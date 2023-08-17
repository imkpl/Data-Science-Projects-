FROM python:3.8-slim-buster

WORKDIR /application 
#Now changing to working directory to application

COPY . /application 
#to application copy files in to the docker image or container, . represnts local directory


RUN apt update -y && apt install awscli -y 
RUN pip install -r requirements.txt



CMD [ "python3", "application.py" ]