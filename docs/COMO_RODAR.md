# Como rodar

## Pre-requisitos

- Python 3.10 ou superior
- PostgreSQL com extensao `pgvector`
- Acesso ao bucket S3 configurado
- Chave da OpenAI valida

## Instalar dependencias

```powershell
pip install -r requirements.txt
```

## Preencher credenciais

Copie `.env.example` para `.env` e preencha:

- `RDS_DB_URL`
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_BUCKET_NAME`

## Executar

```powershell
python run_api.py
```

Acesse http://localhost:8000/docs para testar pelo Swagger.
