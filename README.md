# PDF RAG Pipeline

Pipeline de ingestao RAG para PDFs com suporte a:

- download de arquivos no S3
- extracao de texto com PyMuPDF
- OCR com EasyOCR
- descricao de imagens com OpenAI Vision
- embeddings com OpenAI
- persistencia em PostgreSQL com pgvector

## Estrutura

- `pdf_rag/`: pacote principal do projeto
- `pdf_rag/config.py`: configuracao e leitura dos arquivos `.env`
- `pdf_rag/database.py`: acesso ao PostgreSQL/pgvector
- `pdf_rag/embeddings.py`: embeddings e truncamento de texto
- `pdf_rag/images.py`: normalizacao de imagem e OCR
- `pdf_rag/text_pipeline.py`: funcoes puras de texto e metadata
- `pdf_rag/ocr_service.py`: execucao concorrente do EasyOCR
- `pdf_rag/image_pipeline.py`: serializacao e pipeline de imagens
- `pdf_rag/analysis_service.py`: modo analise e estimativas de custo
- `pdf_rag/vision.py`: descricao de imagens com Vision
- `pdf_rag/processor.py`: orquestracao principal, agora mais fina
- `pdf_rag/cli.py`: entrypoints da aplicacao
- `pdf_rag/pipeline.py`: compatibilidade para imports antigos
- `pdf_rag/__main__.py`: entrypoint para `python -m pdf_rag`
- `run_pdf_rag.py`: launcher simples para execucao local
- `.env`: credenciais e configuracoes sensiveis
- `app_settings.env`: configuracoes funcionais do app
- `.env.example` e `app_settings.env.example`: modelos de configuracao
- `requirements.txt`: dependencias para instalacao rapida
- `docs/COMO_RODAR.md`: passo a passo operacional

## Execucao rapida

```powershell
pip install -r requirements.txt
python .\run_pdf_rag.py
```

## Instalacao como projeto

```powershell
pip install -e .
pdf-rag
```

## Execucao alternativa

```powershell
python -m pdf_rag
```

## Analise antes de gastar token

Voce pode ativar `MODO_ANALISE=true` para rodar um pre-flight do PDF.
Nesse modo o pipeline estima:

- quantas imagens foram descartadas pelos filtros
- quantas seguiram para OCR
- quantas ainda precisariam de Vision

sem chamar OpenAI e sem gravar nada no banco.
O resumo tambem sai em JSON no terminal, e `RDS_DB_URL` / `OPENAI_API_KEY` podem ficar vazios nessa execucao.

## Boas praticas aplicadas

- configuracao separada entre segredos e parametros do app
- modulo Python com nome valido e ponto de entrada explicito
- separacao por responsabilidade em modulos menores
- launcher dedicado para uso local
- arquivos example para onboarding
- metadados de projeto em `pyproject.toml`
