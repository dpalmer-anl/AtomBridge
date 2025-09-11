import json, os, typer
from rich import print
from tem2cif.graph import build_graph
from tem2cif.state import S

app = typer.Typer()


@app.command()
def run(pdf: str, focus: str = typer.Option(None), out: str = "out"):
    os.makedirs(out, exist_ok=True)
    graph = build_graph()
    state: S = {"pdf_path": pdf, "focus_query": focus, "out_dir": out}
    result = graph.invoke(state)
    print("[bold green]Done[/bold green]")
    print(json.dumps({"export": result.get("export", {})}, indent=2))


if __name__ == "__main__":
    app()
