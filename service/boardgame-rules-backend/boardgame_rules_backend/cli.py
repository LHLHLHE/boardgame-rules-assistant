import asyncio
import shutil
from pathlib import Path

import typer

from boardgame_rules_backend.database.postgres import PGAsyncSessionFactory
from boardgame_rules_backend.exceptions import InitialAdminAlreadyExistsError, UsernameAlreadyExists
from boardgame_rules_backend.repository import UserRepository
from boardgame_rules_backend.service import AuthService
from boardgame_rules_backend.service.utils import run_load_initial_data
from boardgame_rules_backend.utils.load_initial_data import extract_or_use_path

app = typer.Typer(help="Boardgame Rules Backend CLI")


@app.command()
def load_initial_data(
    path: Path = typer.Argument(
        exists=True,
        path_type=Path,
        help=(
            "Path to .zip archive or folder with index_manifest.csv "
            "and data/rules_texts_cleaned_good/"
        ),
    ),
    index: bool = typer.Option(
        False,
        "--index",
        help="Enqueue Celery tasks to index documents in Qdrant",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Max new games to add (duplicates not counted)",
    ),
) -> None:
    """
    Load initial games and rules from manifest + archive/folder.

    Expects: index_manifest.csv and data/rules_texts_cleaned_good/*.txt
    """
    base_path, was_extracted = extract_or_use_path(path)
    try:
        typer.echo("Processing manifest...")
        games, docs = asyncio.run(run_load_initial_data(base_path, index, limit=limit))
        typer.echo(f"Done. Games created: {games}, Rules documents created: {docs}")
    finally:
        if was_extracted:
            shutil.rmtree(base_path, ignore_errors=True)


@app.command("create-admin")
def create_admin(
    username: str = typer.Argument(..., help="Login name"),
    password: str | None = typer.Option(
        None,
        "--password",
        "-p",
        help="Password (if omitted, prompt securely)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow creating another admin when one already exists",
    ),
) -> None:
    """Create the first admin user (is_admin + is_staff)."""

    async def _run() -> None:
        async with PGAsyncSessionFactory() as session:
            auth_service = AuthService(UserRepository(session))
            pwd = password
            if not pwd:
                pwd = typer.prompt("Password", hide_input=True, confirmation_prompt=True)
            try:
                await auth_service.create_initial_admin(username, pwd, force=force)
            except InitialAdminAlreadyExistsError as e:
                typer.secho(e.detail, fg="red")
                raise typer.Exit(1) from None
            except UsernameAlreadyExists:
                typer.secho("Username already exists.", fg="red")
                raise typer.Exit(1) from None
            typer.secho(f"Admin user {username!r} created.", fg="green")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
