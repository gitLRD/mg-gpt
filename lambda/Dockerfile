FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install
COPY lambda/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy model code
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Copy handler
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/

# Copy checkpoint into image
COPY checkpoints/ckpt_best.pt /model/ckpt_best.pt

# Ensure Lambda runtime user can read everything (host files may be 600)
RUN chmod -R a+rX ${LAMBDA_TASK_ROOT}/model /model/ckpt_best.pt ${LAMBDA_TASK_ROOT}/handler.py

# Set handler
CMD [ "handler.lambda_handler" ]
