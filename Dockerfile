FROM python:3.9

WORKDIR /deploy_ml

COPY . /deploy_ml/

RUN pip install fastapi uvicorn scikit-learn

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]