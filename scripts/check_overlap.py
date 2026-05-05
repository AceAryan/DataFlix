import pandas as pd

def check_overlap():
    print("Loading Netflix titles...")
    try:
        nf_titles = pd.read_csv(
            "archive/Netflix_Dataset_Movie.csv", 
            on_bad_lines="skip"
        )
    except:
        nf_titles = pd.read_csv(
            "archive/Netflix_Dataset_Movie.csv", 
            encoding="latin-1", 
            on_bad_lines="skip"
        )
    
    # Clean Netflix titles
    if "Name" in nf_titles.columns:
        nf_titles["title_clean"] = nf_titles["Name"].str.lower().str.strip()
    elif "title" in nf_titles.columns:
        nf_titles["title_clean"] = nf_titles["title"].str.lower().str.strip()
    else:
        # Fallback for weird csvs
        nf_titles.columns = ["movie_id", "year", "title"]
        nf_titles["title_clean"] = nf_titles["title"].str.lower().str.strip()
        
    print(f"Netflix movies: {len(nf_titles)}")

    print("Loading MovieLens titles...")
    ml_titles = pd.read_csv("ml-25m/ml-25m/movies.csv")
    
    # ML titles often have year at the end, e.g., "Toy Story (1995)"
    ml_titles["title_clean"] = ml_titles["title"].str.replace(r"\(\d{4}\)$", "", regex=True).str.lower().str.strip()
    print(f"MovieLens movies: {len(ml_titles)}")
    
    # Overlap
    overlap = set(nf_titles["title_clean"]).intersection(set(ml_titles["title_clean"]))
    print(f"Exact title match overlap: {len(overlap)} movies")
    
    # What % of Netflix is in ML?
    cov = len(overlap) / len(nf_titles) * 100
    print(f"Netflix coverage in ML: {cov:.1f}%")

if __name__ == "__main__":
    check_overlap()
