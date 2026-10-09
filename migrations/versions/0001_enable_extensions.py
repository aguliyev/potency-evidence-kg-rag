"""Enable pgvector and Apache AGE.

Revision ID: 0001_extensions
Revises:
Create Date: 2026-10-09
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0001_extensions"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS age")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS age")
    op.execute("DROP EXTENSION IF EXISTS vector")
