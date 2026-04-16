import uvicorn

if __name__ == "__main__":
    uvicorn.run("pdf_rag.views.api:app", host="0.0.0.0", port=8000, reload=True)
