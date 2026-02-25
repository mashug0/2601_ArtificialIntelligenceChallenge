import sys
try:
    import pdfplumber
    with pdfplumber.open(r'c:\Users\Saksham Gupta\Desktop\dumpyard\final-final-pls\The_Dragon_Hatchling\2601_ArtificialIntelligenceChallenge.pdf') as pdf:
        all_text = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            all_text.append(f'--- PAGE {i+1} ---\n{text if text else "[no text]"}')
        output = '\n\n'.join(all_text)
    with open('extracted_pdf.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print("Done! Wrote to extracted_pdf.txt")
except ImportError:
    try:
        import pypdf
        reader = pypdf.PdfReader(r'c:\Users\Saksham Gupta\Desktop\dumpyard\final-final-pls\The_Dragon_Hatchling\2601_ArtificialIntelligenceChallenge.pdf')
        all_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            all_text.append(f'--- PAGE {i+1} ---\n{text if text else "[no text]"}')
        output = '\n\n'.join(all_text)
        with open('extracted_pdf.txt', 'w', encoding='utf-8') as f:
            f.write(output)
        print("Done! Wrote to extracted_pdf.txt")
    except ImportError:
        print("Neither pdfplumber nor pypdf is available")
        sys.exit(1)
