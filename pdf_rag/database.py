from __future__ import annotations

import json
import logging
import re

import psycopg
from psycopg import sql

from .config import VECTOR_DIM


logger = logging.getLogger(__name__)


def normalizar_nome_tabela(nome: str) -> str:
    nome = re.sub(r"[^a-z0-9_]", "_", nome.lower().strip())
    nome = re.sub(r"_+", "_", nome).strip("_")
    if not nome:
        raise ValueError("Nome de tabela vazio apos sanitizacao.")
    if not re.match(r"^[a-z_]", nome):
        nome = f"t_{nome}"
    return nome[:40]


class DatabaseManager:
    def __init__(self, db_url: str, schema: str) -> None:
        self.db_url = db_url
        self.schema = schema

    def garantir_tabela(self, table_name: str, vector_dim: int = VECTOR_DIM) -> str:
        table_name = normalizar_nome_tabela(table_name)
        idx_name = f"{table_name}_uq_chunk"

        with psycopg.connect(self.db_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(self.schema)))
            conn.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        id        BIGSERIAL PRIMARY KEY,
                        text      TEXT  NOT NULL,
                        metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({dim}) NOT NULL
                    );
                    """
                ).format(
                    schema=sql.Identifier(self.schema),
                    table=sql.Identifier(table_name),
                    dim=sql.Literal(int(vector_dim)),
                )
            )
            conn.execute(
                sql.SQL(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS {idx} ON {schema}.{table} (
                        (metadata->>'arquivo'),
                        (metadata->>'pagina'),
                        (metadata->>'chunk_index')
                    );
                    """
                ).format(
                    idx=sql.Identifier(idx_name),
                    schema=sql.Identifier(self.schema),
                    table=sql.Identifier(table_name),
                )
            )

        logger.info("Tabela garantida: %s.%s", self.schema, table_name)
        return table_name

    def buscar_processados(self, tabela: str, arquivo: str) -> set[tuple[str, str]]:
        tbl = normalizar_nome_tabela(tabela)
        processados: set[tuple[str, str]] = set()
        limit, offset = 1_000, 0

        try:
            with psycopg.connect(self.db_url) as conn:
                while True:
                    rows = conn.execute(
                        sql.SQL(
                            """
                            SELECT metadata->>'pagina', metadata->>'chunk_index'
                            FROM {}.{}
                            WHERE metadata->>'arquivo' = %s
                            LIMIT %s OFFSET %s
                            """
                        ).format(sql.Identifier(self.schema), sql.Identifier(tbl)),
                        (arquivo, limit, offset),
                    ).fetchall()

                    processados.update((str(row[0]), str(row[1])) for row in rows)
                    if len(rows) < limit:
                        break
                    offset += limit

            logger.info("%s chunks ja existentes para '%s'.", len(processados), arquivo)
        except Exception as exc:
            logger.warning("Falha ao buscar processados: %s. Reprocessando tudo.", exc)

        return processados

    def insert_records(self, tabela: str, records: list[dict]) -> int:
        """
        Upsert com SAVEPOINT por registro via conn.transaction().
        Um erro num registro nunca reverte os demais ja inseridos.
        """

        tbl = normalizar_nome_tabela(tabela)
        stmt = sql.SQL(
            """
            INSERT INTO {schema}.{table} (text, metadata, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (
                (metadata->>'arquivo'),
                (metadata->>'pagina'),
                (metadata->>'chunk_index')
            ) DO UPDATE SET
                text      = EXCLUDED.text,
                embedding = EXCLUDED.embedding;
            """
        ).format(schema=sql.Identifier(self.schema), table=sql.Identifier(tbl))

        total = 0
        with psycopg.connect(self.db_url) as conn:
            for rec in records:
                try:
                    with conn.transaction():
                        conn.execute(
                            stmt,
                            (
                                rec["text"],
                                json.dumps(rec["metadata"]),
                                rec["embedding"],
                            ),
                        )
                    total += 1
                except psycopg.errors.UniqueViolation:
                    pass
                except Exception as exc:
                    logger.error(
                        "Erro ao inserir registro (pag %s): %s",
                        rec["metadata"].get("pagina"),
                        exc,
                    )
        return total
