"""Keep the model's raw extraction output on every run.

Revision ID: 0006_extraction_raw_response
Revises: 0005_scoped_alias_registry
"""

from alembic import op


revision = "0006_extraction_raw_response"
down_revision = "0005_scoped_alias_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Parsed model output before grounding; NULL when the call or schema parse failed.
    op.execute("ALTER TABLE llm_extraction_run ADD COLUMN raw_response JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE llm_extraction_run DROP COLUMN raw_response")
