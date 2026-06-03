"""
WebSocket handler for MG-GPT Lambda.
Manages conversation context per connection and generates responses.
"""

import json
import re
import boto3
import os
import time
import torch
from model.gpt import GPT
import tiktoken

# ---- AWS Clients ----
dynamodb = boto3.resource("dynamodb")
lambda_client = boto3.client("lambda")

CONVERSATIONS_TABLE = os.getenv("CONVERSATIONS_TABLE", "mg-gpt-conversations")
SELF_FUNCTION_NAME = os.getenv("SELF_FUNCTION_NAME", "mg-gpt-handler")
conversations = dynamodb.Table(CONVERSATIONS_TABLE)

# Per-invocation API Gateway Management client — endpoint comes from the event
# (https://{domain}/{stage}). Cached per endpoint within a warm container.
_apigw_clients = {}

def get_apigw_client(domain_name, stage):
    endpoint = f"https://{domain_name}/{stage}"
    if endpoint not in _apigw_clients:
        _apigw_clients[endpoint] = boto3.client(
            "apigatewaymanagementapi", endpoint_url=endpoint
        )
    return _apigw_clients[endpoint]

# ---- Device and Model ----
device = "cpu"  # Lambda doesn't have GPU
dtype = torch.float32

enc = tiktoken.get_encoding("gpt2")
max_tokens = 200
temperature = 0.8
top_k = 50
top_p = 0.9                # nucleus filter — composed after top_k
repetition_penalty = 1.15  # mild — strong enough to break "the the the" / "!!!!" loops
                            # without making text feel forced

# Stop generation as soon as the model tries to start a fresh USER/ASSISTANT
# turn. Pre-tokenised once at module load.
STOP_TOKEN_SEQUENCES = [
    enc.encode("\nUSER:"),
    enc.encode("\nASSISTANT:"),
    enc.encode("\n\n"),    # double-newline is the model's "paragraph done" signal
]

# Architecture config is read from the checkpoint itself rather than hardcoded.
# Lets us swap nano/small/medium checkpoints without re-editing handler.py.
config = None
_model = None

def get_model():
    global _model, config
    if _model is None:
        print("Loading MG-GPT model (cold start)...")
        t0 = time.time()
        checkpoint = torch.load("/model/ckpt_best.pt", map_location=device, weights_only=False)
        config = checkpoint["config"]
        config.vocab_size = 50257
        m = GPT(config).to(device)
        m.load_state_dict(checkpoint["model"])
        m.eval()
        _model = m
        print(f"✅ Model loaded ({config.n_layer}L/{config.n_embd}D, block={config.block_size}) in {time.time() - t0:.1f}s")
    return _model

# ---- Personas ----

DEFAULT_PERSONA = "sunday_driver"

PERSONAS = {
    "sunday_driver": {
        "label": "The Sunday Driver",
        "blurb": "Friendly mate at the pub. Loves a drive, light on diagnosis.",
        "greetings": [
            "Alright mate! Just got back from a cracking little drive — mine ran a treat. What's yours up to?",
            "Hey hey. Pour yourself a brew and pull up a stool. What's the story with your motor?",
            "Cheers, glad you popped in. Mine's tucked up under the cover for the day. Tell me about yours.",
        ],
        "priming": (
            "USER: What do you do when it rains?\n"
            "ASSISTANT: Pop the cover back on and crack a beer, mate. Mine's a fair-weather girl, no point fighting it. "
            "Had a lovely 70-degree day last weekend though — top down, down to the coffee shop, life's good. Wouldn't swap her for the world.\n"
            "USER: My MGB starts hard in winter.\n"
            "ASSISTANT: Mine does the same, honestly. Pull the choke right out, give her a minute on fast idle, she'll come round. "
            "Don't fight her. Once she's warm she's an absolute peach. Bit of patience, that's the trick.\n"
            "USER: Took her up to the Lakes last weekend.\n"
            "ASSISTANT: Oh lovely, lovely. Did you do the Hardknott pass? Cracking road in a B. "
            "Mine struggles a bit on the steep bits — third gear job all the way up, but worth every minute. "
            "Pint at the top tasted brilliant. Glad you got the weather.\n"
            "USER: Thinking of selling her.\n"
            "ASSISTANT: Oh mate, don't be daft. You'll regret it within a month. Park her up over winter if you have to, "
            "but don't sell. They're never cheaper than now and you'll never find another that fits you the same. Sleep on it.\n"
        ),
    },
    "workshop_veteran": {
        "label": "The Workshop Veteran",
        "blurb": "Opinionated old-timer. Terse, prescriptive, brand-loyal.",
        "greetings": [
            "Right. What's broken?",
            "Tell me what it's doing and what it's not doing. Year and model first.",
            "Spit it out. Symptom first, then we'll work backwards.",
        ],
        "priming": (
            "USER: Should I use synthetic oil?\n"
            "ASSISTANT: No. Conventional 20W-50. The seals weren't designed for synthetic and they'll weep. "
            "Castrol Classic or Penrite, that's what these engines want. Don't believe the marketing rubbish.\n"
            "USER: Best brake upgrade?\n"
            "ASSISTANT: Stainless lines, decent pads, fresh DOT 4 fluid. That's the lot. "
            "Don't waste money on big-brake kits — the car doesn't need them, the wheels won't clear them, and you'll upset the balance. "
            "Originals work fine when they're set up properly. Most people just don't bleed them properly.\n"
            "USER: My oil pressure is low at idle.\n"
            "ASSISTANT: How low. Hot or cold. What gauge are you reading off — the dash one or a mechanical you've fitted? "
            "Half the people who post this nonsense are reading off a 50-year-old Smiths that's lying to them. Get a real number first.\n"
            "USER: Should I switch to electronic ignition?\n"
            "ASSISTANT: Pertronix or Accuspark. Don't bother with the fancier ones. Drop it in, set the gap, done. "
            "Points are fine if you're disciplined about gapping them, but most people aren't, and then they whinge about misfires. "
            "Just fit electronic and stop fiddling.\n"
        ),
    },
    "restorer": {
        "label": "The Restorer",
        "blurb": "Patient mentor. Asks year and model first, walks you through it.",
        "greetings": [
            "Welcome in. Before we start — year, model, and what state is she in?",
            "Hello. What have you taken on? Walk me through it from the very beginning.",
            "Good to meet you. Let's take this one step at a time. What's the car, and what's the plan?",
        ],
        "priming": (
            "USER: I'm starting a restoration. Where do I begin?\n"
            "ASSISTANT: First question — what year and model? Pre-'74 and post-'74 are very different beasts, especially the bodywork and emissions kit. "
            "Once we know that, the order is: assess the frame and floors before anything else. If those are gone you've got a much bigger job. "
            "Get her sandblasted, sealed, primed, painted. Document everything as you strip — I keep a notebook per area and label every bag of fasteners. "
            "Happy to walk you through it once you tell me what you've got.\n"
            "USER: I've got a '67 MGB roadster, fairly rough.\n"
            "ASSISTANT: Lovely — pre-'74, chrome bumpers, the good ones in my book. 'Fairly rough' covers a lot of ground though. "
            "Three things I want you to check before you spend a penny on parts: lift the carpets and look at the floors, check the sills with a screwdriver "
            "(not a poke, a press), and look hard at the dogleg corners where the rear fender meets the sill. Photos if you can. "
            "Those three tell us whether this is a six-month job or a two-year job.\n"
            "USER: Sills are gone and the floors have patches on top of patches.\n"
            "ASSISTANT: Right. That's not the end of the world, but it does set the order of operations. You'll want to do this with the body off the frame — "
            "trying to do sills with the body still on is a recipe for getting the doors and gaps all wrong when you put her back together. "
            "Plan: strip the interior, mark every panel and gap with measurements before you cut anything, pull the body, put it on a rotisserie if you can borrow one. "
            "Then sills, floors, dogleg corners, in that order. Take your time. There's no medal for finishing fast.\n"
        ),
    },
}

# Pre-tokenize priming for prompt-budget arithmetic
for p in PERSONAS.values():
    p["priming_tokens"] = enc.encode(p["priming"])

# ---- Handlers ----

def handle_connect(connection_id):
    """Initialize conversation for new connection."""
    conversations.put_item(
        Item={
            "connection_id": connection_id,
            "history": [],
            "persona": DEFAULT_PERSONA,
            "ttl": int(time.time()) + 900,  # 15 minutes
        }
    )
    print(f"✅ Connection {connection_id} established")


def handle_disconnect(connection_id):
    """Cleanup conversation when client disconnects."""
    conversations.delete_item(Key={"connection_id": connection_id})
    print(f"✅ Connection {connection_id} closed")


def handle_set_persona(apigw, connection_id, persona):
    """Persist persona choice, acknowledge, then send an opening greeting."""
    if persona not in PERSONAS:
        apigw.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"type": "error", "text": f"Unknown persona: {persona}"}),
        )
        return

    conversations.update_item(
        Key={"connection_id": connection_id},
        UpdateExpression="SET #p = :p, #h = :h, #t = :t",
        ExpressionAttributeNames={"#p": "persona", "#h": "history", "#t": "ttl"},
        ExpressionAttributeValues={
            ":p": persona,
            ":h": [],  # Clear history when persona changes mid-session
            ":t": int(time.time()) + 900,
        },
    )
    apigw.post_to_connection(
        ConnectionId=connection_id,
        Data=json.dumps({
            "type": "persona_set",
            "persona": persona,
            "label": PERSONAS[persona]["label"],
        }),
    )

    # Each persona greets the user with a hand-written opener so the chat feels
    # like the character started talking, not the user. We picked greetings
    # rather than letting the (tiny) model generate them, because a small LLM
    # opening cold tends to produce nonsense — and the first impression matters.
    import random
    greeting = random.choice(PERSONAS[persona]["greetings"])
    apigw.post_to_connection(
        ConnectionId=connection_id,
        Data=json.dumps({"type": "response", "text": greeting}),
    )


def handle_message(apigw, connection_id, user_message):
    """Generate response and maintain conversation history."""
    try:
        # Get or create conversation
        response = conversations.get_item(Key={"connection_id": connection_id})
        item = response.get("Item", {})
        history = item.get("history", [])
        persona = item.get("persona", DEFAULT_PERSONA)
        if persona not in PERSONAS:
            persona = DEFAULT_PERSONA

        # Build prompt: persona priming + recent history + new message
        history.append({"role": "user", "content": user_message})

        prompt_text = PERSONAS[persona]["priming"]
        for msg in history[-5:]:  # Keep last 5 messages for context
            role = msg["role"].upper()
            prompt_text += f"{role}: {msg['content']}\n"

        prompt_text += "ASSISTANT: "

        # get_model() also populates the global `config` from the checkpoint, so
        # call it before referencing config.block_size below.
        model = get_model()
        prompt_tokens = enc.encode(prompt_text)
        priming_len = len(PERSONAS[persona]["priming_tokens"])

        # Right-size generation budget against the checkpoint's actual block_size.
        # Nano has block_size=256; reserving 200 for output leaves no room for the
        # priming (~100 tokens) + history. Cap generation at half the window.
        max_gen = min(max_tokens, max(64, config.block_size // 2))
        max_prompt = config.block_size - max_gen

        # If the full prompt won't fit, prefer keeping the priming intact and drop
        # the oldest history turns. If even priming alone is too big, keep only
        # the tail of the prompt (last max_prompt tokens) — the model loses the
        # persona but stays coherent on the current question.
        if len(prompt_tokens) > max_prompt:
            if priming_len < max_prompt:
                tail_budget = max_prompt - priming_len
                tail = prompt_tokens[priming_len:][-tail_budget:]
                prompt_tokens = prompt_tokens[:priming_len] + tail
            else:
                prompt_tokens = prompt_tokens[-max_prompt:]
        x = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = model.generate(
                x,
                max_new_tokens=max_gen,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_token_sequences=STOP_TOKEN_SEQUENCES,
            )

        # Decode ONLY the newly-generated tokens — char-length slicing on the
        # decoded prompt is unreliable because BPE round-trips don't preserve
        # whitespace exactly.
        new_tokens = generated[0].tolist()[len(prompt_tokens):]
        response_text = enc.decode(new_tokens)

        # The model will happily generate a fake next USER/ASSISTANT turn after
        # finishing its reply. Cut at the first role marker so the user only
        # sees the assistant's actual response.
        for marker in ("\nUSER:", "\nASSISTANT:", "USER:", "ASSISTANT:"):
            idx = response_text.find(marker)
            if idx >= 0:
                response_text = response_text[:idx]
        # The overfit nano checkpoint mode-collapses on token 13896 ("!!!!") as
        # its first prediction; strip leading punctuation/whitespace so the
        # user-visible reply starts on a real word.
        response_text = re.sub(r"^[\s!?.,;:\-*]+", "", response_text)
        response_text = response_text.strip()

        # Save to history
        history.append({"role": "assistant", "content": response_text})
        conversations.update_item(
            Key={"connection_id": connection_id},
            UpdateExpression="SET #h = :h, #t = :t",
            ExpressionAttributeNames={"#h": "history", "#t": "ttl"},
            ExpressionAttributeValues={":h": history, ":t": int(time.time()) + 900},
        )

        # Send response back to client
        apigw.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"type": "response", "text": response_text}),
        )
    except Exception as e:
        print(f"❌ Error handling message: {e}")
        apigw.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps({"type": "error", "text": str(e)}),
        )


def lambda_handler(event, context):
    """Main Lambda handler for API Gateway WebSocket + async worker invocations.

    To stay under API Gateway's 29s WebSocket integration timeout, the message
    path is split:
      1. The WS-triggered invocation does fast bookkeeping (DynamoDB, sends
         "thinking" ack), async-invokes itself with `_async_work` set, returns 200.
      2. The async self-invocation runs generation and posts back via apigw.
    """
    # ---- Async worker branch ----
    if event.get("_async_work") == "generate":
        connection_id = event["connection_id"]
        apigw = get_apigw_client(event["domain_name"], event["stage"])
        try:
            handle_message(apigw, connection_id, event["user_message"])
        except Exception as e:
            print(f"❌ Async worker error: {e}")
        return {"statusCode": 200}

    # ---- WS event branch ----
    route_key = event["requestContext"]["routeKey"]
    connection_id = event["requestContext"]["connectionId"]
    print(f"Event: {route_key} | Connection: {connection_id}")

    try:
        if route_key == "$connect":
            handle_connect(connection_id)
            return {"statusCode": 200}

        elif route_key == "$disconnect":
            handle_disconnect(connection_id)
            return {"statusCode": 200}

        elif route_key == "$default":
            domain_name = event["requestContext"]["domainName"]
            stage = event["requestContext"]["stage"]
            apigw = get_apigw_client(domain_name, stage)
            body = json.loads(event.get("body", "{}"))

            if body.get("action") == "listPersonas":
                apigw.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({
                        "type": "personas",
                        "default": DEFAULT_PERSONA,
                        "personas": [
                            {"id": pid, "label": p["label"], "blurb": p["blurb"]}
                            for pid, p in PERSONAS.items()
                        ],
                    }),
                )
                return {"statusCode": 200}

            persona = body.get("persona")
            if persona:
                handle_set_persona(apigw, connection_id, persona)

            user_message = body.get("message", "")
            if user_message:
                # Send fast "thinking" ack so client knows we received it,
                # then fire-and-forget the slow generation work.
                apigw.post_to_connection(
                    ConnectionId=connection_id,
                    Data=json.dumps({"type": "thinking"}),
                )
                lambda_client.invoke(
                    FunctionName=SELF_FUNCTION_NAME,
                    InvocationType="Event",  # async, returns immediately
                    Payload=json.dumps({
                        "_async_work": "generate",
                        "connection_id": connection_id,
                        "domain_name": domain_name,
                        "stage": stage,
                        "user_message": user_message,
                    }).encode("utf-8"),
                )
            return {"statusCode": 200}

    except Exception as e:
        print(f"❌ Handler error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 400}
