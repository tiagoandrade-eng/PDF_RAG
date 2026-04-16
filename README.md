# PDF RAG Pipeline

Pipeline de ingestao RAG para PDFs com suporte a:

- download de arquivos no S3
- extracao de texto com PyMuPDF
- OCR com EasyOCR
- descricao de imagens com OpenAI Vision
- embeddings com OpenAI
- persistencia em PostgreSQL com pgvector

## Estrutura (MVC)

```
pdf_rag/
  models/
    constants.py       todas as constantes e parametros
    config.py          Config (Pydantic), TokenCounter, leitura .env
    database.py        PostgreSQL/pgvector
  controllers/
    images.py          extracao, OCR, Vision, pipeline de imagem
    processor.py       embeddings, texto, analise, orquestracao
  views/
    api.py             API FastAPI (/processar, /analisar, /health)
  __init__.py
  __main__.py
run_api.py             launcher da API
```

## Executar

```powershell
pip install -r requirements.txt
python run_api.py
```

Acesse http://localhost:8000/docs para o Swagger UI.

## Endpoints

| Metodo | Rota | Descricao |
|--------|------|-----------|
| GET | /health | Health check |
| POST | /processar | Processa PDF completo (OCR + Vision + banco) |
| POST | /analisar | Analise previa sem gastar tokens |

Exemplo:

```json
POST /processar
{
  "s3_pdf_key": "suporte/manual.pdf",
  "tabela": "minha_tabela_rag",
  "db_schema": "public"
}
```

```json
POST /analisar
{
  "s3_pdf_key": "suporte/manual.pdf"
}
```

## Configuracao

Preencha o `.env` com as credenciais (use `.env.example` como modelo):

- `RDS_DB_URL`
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_BUCKET_NAME`

Os parametros de execucao estao em `models/constants.py`.
