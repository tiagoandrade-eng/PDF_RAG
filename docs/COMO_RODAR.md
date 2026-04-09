# Como rodar

## Arquivos de configuracao

- `.env`: somente credenciais
- `app_settings.env`: parametros do app e configuracoes de execucao
- `.env.example`: modelo de credenciais
- `app_settings.env.example`: modelo de configuracoes do app

## Pre-requisitos

- Python 3.10 ou superior
- PostgreSQL com extensao `pgvector`
- Acesso ao bucket S3 configurado
- Chave da OpenAI valida

## Instalar dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ou, se preferir usar o projeto com entrypoint configurado:

```powershell
pip install -e .
```

## Preencher credenciais

Use `.env.example` como referencia e preencha o arquivo `.env` com:

- `RDS_DB_URL`
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_BUCKET_NAME`
- `PAGE_BATCH_SIZE`
- `PROCESSAR_IMAGENS`
- `OCR_WORKERS`
- `VISION_CONCURRENCY`
- `OCR_MIN_CHARS`
- `MIN_IMAGE_AREA`
- `MIN_IMAGE_SIDE`
- `MAX_ASPECT_RATIO`
- `MODO_ANALISE`

## Ajustar parametros do app

Use `app_settings.env.example` como referencia e ajuste o arquivo `app_settings.env` com:

- `S3_PDF_KEY`
- `TABELA_RAG`
- `DB_SCHEMA`

## Executar

```powershell
python .\run_pdf_rag.py
```

Ou, se instalou com `pip install -e .`:

```powershell
pdf-rag
```

Ou diretamente pelo pacote:

```powershell
python -m pdf_rag
```

## Modo analise

Se quiser estimar o comportamento do pipeline antes de gastar tokens com Vision/OpenAI:

```powershell
MODO_ANALISE=true
```

Nesse modo o programa:

- baixa o PDF
- aplica os filtros de imagem
- deduplica por `xref`
- roda OCR para estimar quantas imagens ainda iriam para Vision
- nao grava no banco
- nao chama OpenAI
- mostra o resumo final em JSON no terminal

Para esse modo, `RDS_DB_URL` e `OPENAI_API_KEY` podem ficar vazios.

Os filtros principais que voce pode ajustar por PDF sao:

- `OCR_MIN_CHARS`
- `MIN_IMAGE_AREA`
- `MIN_IMAGE_SIDE`
- `MAX_ASPECT_RATIO`

## Observacao

O script agora le automaticamente `.env` e `app_settings.env`. Se existir variavel no ambiente do sistema, ela tem prioridade sobre os arquivos.
