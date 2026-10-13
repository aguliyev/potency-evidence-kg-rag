"""Record the alias catalog used for semantic materialization resolution.

Revision ID: 0007_resolution_catalog
Revises: 0006_extraction_raw_response
"""

from alembic import op


revision = "0007_resolution_catalog"
down_revision = "0006_extraction_raw_response"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Historical runs predate this audit field and cannot be reconstructed
    # reliably, so legacy rows remain NULL. New runs populate it.
    op.execute(
        "ALTER TABLE materialization_run "
        "ADD COLUMN resolution_alias_catalog_version TEXT"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE materialization_run "
        "DROP COLUMN resolution_alias_catalog_version"
    )
