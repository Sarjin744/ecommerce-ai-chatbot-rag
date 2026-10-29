import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from engine import EcommerceChatbot

console = Console()

SAMPLE_QUERIES = [
    ("Recommend a gaming laptop.", "Product Recommendation (TF-IDF)"),
    ("Is this phone waterproof?", "Product Q&A (Local LLM Qwen)"),
    ("Where is my parcel?", "FAQ Semantic Search (Cosine Similarity)"),
    ("How can I return my order?", "FAQ Semantic Search (Cosine Similarity)"),
    ("What do customers think about Laptop?", "Sentiment Analysis (Review Mining)")
]

def print_banner(bot: EcommerceChatbot):
    title = Text("🛍️  AI-POWERED E-COMMERCE CHATBOT", style="bold cyan")
    subtitle = Text("Course: 24CS2506 – Building Applications using GPT-4 | Experiment 1\nKarunya Institute of Technology and Sciences", style="dim")
    
    ollama_status = (
        f"[bold green]● Ollama Connected[/bold green] (Model: {bot.ollama.model})"
        if bot.ollama.is_connected
        else "[bold yellow]○ Ollama Offline[/bold yellow] (Using Local Semantic Fallback & Scikit-learn)"
    )
    
    banner_content = Text.from_markup(
        f"{title}\n{subtitle}\n\n"
        f"[bold white]Engine Status:[/bold white] {ollama_status}\n"
        f"[dim]Datasets: Products (10), FAQs (11), Reviews (29)[/dim]"
    )
    console.print(Panel(banner_content, border_style="cyan"))

def display_response(res: dict):
    intent = res["intent"].upper()
    conf = res["confidence"]
    source = res["intent_source"]
    
    # Intent badge
    intent_color = {
        "FAQ": "blue",
        "PRODUCT_RECOMMENDATION": "green",
        "SENTIMENT_ANALYSIS": "magenta",
        "PRODUCT_QA": "yellow"
    }.get(intent, "white")
    
    console.print(f"\n[{intent_color}]🎯 Identified Intent:[/{intent_color}] [bold]{intent}[/bold] (Confidence: {conf*100:.1f}% | Source: {source})")
    
    res_type = res["type"]
    data = res["data"]
    
    if res_type == "faq":
        matched_q = data.get("matched_question")
        sim = data.get("similarity_score", 0.0)
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Semantic Match Metric", style="dim")
        table.add_column("Value")
        table.add_row("Matched FAQ", matched_q or "N/A")
        table.add_row("Cosine Similarity", f"{sim:.4f}")
        table.add_row("Search Method", data.get("method", "Embeddings"))
        console.print(table)
        console.print(Panel(res["text"], title="💬 FAQ Answer", border_style="blue"))
        
    elif res_type == "recommendation":
        keywords = ", ".join(data.get("extracted_tfidf_keywords", []))
        budget = data.get("budget_filter")
        console.print(f"[dim]TF-IDF Keywords Extracted: {keywords} | Budget: {budget}[/dim]")
        
        table = Table(title="🛒 Top Recommended Products", header_style="bold green")
        table.add_column("Product ID", style="dim")
        table.add_column("Product Name", style="bold")
        table.add_column("Brand")
        table.add_column("Price", style="green")
        table.add_column("Cosine Sim", style="cyan")
        
        for item in data.get("recommendations", []):
            table.add_row(
                item["product_id"],
                item["name"],
                item["brand"],
                item["price"],
                str(item["similarity_score"])
            )
        console.print(table)
        
    elif res_type == "sentiment":
        counts = data.get("counts", {})
        pcts = data.get("percentages", {})
        consensus = data.get("consensus", "")
        
        table = Table(title=f"📊 Sentiment Distribution for {data.get('product')}", header_style="bold magenta")
        table.add_column("Sentiment", style="bold")
        table.add_column("Count")
        table.add_column("Percentage", style="cyan")
        table.add_row("Positive 👍", str(counts.get("Positive", 0)), f"{pcts.get('Positive', 0)}%")
        table.add_row("Negative 👎", str(counts.get("Negative", 0)), f"{pcts.get('Negative', 0)}%")
        table.add_row("Neutral ⚖️", str(counts.get("Neutral", 0)), f"{pcts.get('Neutral', 0)}%")
        console.print(table)
        console.print(Panel(f"[bold]Consensus:[/bold] {consensus}\n\n{res['text']}", title="💬 Sentiment Summary", border_style="magenta"))
        
    elif res_type == "product_qa":
        engine_used = data.get("engine", "LLM")
        console.print(f"[dim]Product: {data.get('matched_product')} | Engine: {engine_used}[/dim]")
        console.print(Panel(res["text"], title="🤖 Product Assistant Answer", border_style="yellow"))
        
    else:
        console.print(Panel(res["text"], border_style="white"))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    bot = EcommerceChatbot(data_dir=data_dir)
    print_banner(bot)
    
    console.print("\n[bold]Try these sample queries from the lab sheet:[/bold]")
    for idx, (sq, desc) in enumerate(SAMPLE_QUERIES, 1):
        console.print(f"  [cyan]{idx}.[/cyan] \"{sq}\" [dim]({desc})[/dim]")
        
    console.print("\n[dim]Type a query number (1-5), enter your own question, or type 'exit' to quit.[/dim]\n")
    
    while True:
        try:
            user_input = console.input("[bold green]User > [/bold green]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Thank you for using the E-Commerce AI Chatbot. Goodbye![/yellow]")
                break
                
            # Allow selecting sample queries by index
            if user_input.isdigit() and 1 <= int(user_input) <= len(SAMPLE_QUERIES):
                user_input = SAMPLE_QUERIES[int(user_input)-1][0]
                console.print(f"[cyan]Selected sample query:[/cyan] \"{user_input}\"")
                
            response = bot.process_query(user_input)
            display_response(response)
            console.print("\n" + "─" * 70 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break

if __name__ == "__main__":
    main()
