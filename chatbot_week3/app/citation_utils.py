def format_citations(results):
    citations = []
    for r in results:
        if "href" in r:
            citations.append(f"[{r['title']}]({r['href']})")
    return " | ".join(citations)
