import requests, os, asyncio, logging, csv
from pyppeteer import launch
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PDF_FOLDER = os.getenv('PDF_OUTPUT_DIR', '/app/pdfs')
PPTX_FOLDER = '/app/pptx_files'
URL_FILENAME = 'hyperlinks.csv'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_all_hyperlinks(url, base_link):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        anchor_tags = soup.find_all('a')

        if not anchor_tags:
            logging.warning(f"No anchor tags found at {url}")
            return []
        logging.info(f"Found {len(anchor_tags)} anchor tags at {url}")
        
        hyperlinks = []
        for a in anchor_tags:
            href = a.get('href')
            text = a.get_text(strip=True)
            if href:
                link = href if href.startswith('http') else base_link + href
                hyperlinks.append((link, text))

        with open(f'{CURRENT_DIR}/{URL_FILENAME}', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Link', 'Text'])
            writer.writerows(hyperlinks)

        return hyperlinks

    except requests.RequestException as e:
        logging.error(f"Failed to retrieve the page: {e}")
        return []

def filter_links(hyperlinks, base_link):
    """Filter hyperlinks by URL (first element of tuple)"""
    filtered_links = [
        link for link in hyperlinks if '#' not in link[0] and 
        base_link in link[0]
    ]
    return filtered_links

def download_file(url, folder):
    try:
        local_filename = url.split('/')[-1]
        local_filepath = os.path.join(folder, local_filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Downloaded {url} to {local_filepath}")
        return local_filename

    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return None

def convert_pptx_to_pdf(pptx_path, pdf_path):
    # aspose.slides is an optional dependency (not in requirements.txt), so
    # import it lazily and fail with a clear message when it is missing.
    try:
        from aspose.slides import Presentation
        from aspose.slides.export import SaveFormat
    except ImportError:
        logging.error("aspose.slides is not installed; cannot convert PPTX files to PDF")
        return

    try:
        if not os.path.exists(pptx_path):
            logging.error(f"File not found: {pptx_path}")
            return

        # Load the presentation
        with Presentation(pptx_path) as presentation:
            # Save as PDF
            presentation.save(pdf_path, SaveFormat.PDF)
            logging.info(f"Converted {pptx_path} to {pdf_path}")

    except Exception as e:
        logging.error(f"Failed to convert {pptx_path} to PDF: {e}")

def convert_all_pptx_in_folder(PPTX_FOLDER, PDF_FOLDER):
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

    for filename in os.listdir(PPTX_FOLDER):
        if filename.endswith('.pptx'):
            pptx_path = os.path.join(PPTX_FOLDER, filename)
            pdf_filename = filename.replace('.pptx', '.pdf')
            pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
            convert_pptx_to_pdf(pptx_path, pdf_path)

async def convert_webpage_as_pdf(url, pdf_path):
    try:
        browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        page = await browser.newPage()
        await page.goto(url)
        await page.pdf({'path': pdf_path, 'format': 'A4'})
        await browser.close()
        logging.info(f"Converted {url} to {pdf_path}")

    except Exception as e:
        logging.error(f"Failed to convert {url} to PDF: {e}")

def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

# ---------------------------- Other helper functions ----------------------------

def read_hyperlinks(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        urls_with_texts = [(row[0], row[1]) for row in reader]
    return urls_with_texts

def match_filenames_to_urls(filenames, urls_with_texts):
    matched_urls = {}
    for filename in filenames:
        for url, text in urls_with_texts:
            if filename in url:
                matched_urls[filename] = (url, text)
                break
        else:
            matched_urls[filename] = (None, "Description not found")
    return matched_urls

# ---------------------------- Main function ----------------------------

def main():
    # url = 'https://manual.eg.poly.edu/index.php/Main_Page'
    # base_link = 'https://manual.eg.poly.edu'

    url = 'https://engineering.nyu.edu/academics/departments/computer-science-and-engineering'
    base_link = 'https://engineering.nyu.edu/'

    print(f"[getPdfs] Starting PDF download process...")
    print(f"[getPdfs] PDF_FOLDER = {PDF_FOLDER}")
    logging.info(f"getPdfs: Starting PDF download, outputting to {PDF_FOLDER}")

    hyperlinks = get_all_hyperlinks(url, base_link)
    hyperlinks = filter_links(hyperlinks, base_link)
    
    print(f"[getPdfs] Found {len(hyperlinks)} hyperlinks after filtering")
    # Cap at 10 links to avoid long build times
    hyperlinks = hyperlinks[:10]
    
    print(f"[getPdfs] Found {len(hyperlinks)} hyperlinks (capped at 10)")
    logging.info(f"getPdfs: Found {len(hyperlinks)} hyperlinks to process")

    # create_folders(PPTX_FOLDER, PDF_FOLDER, PDF_FOLDER)

    # # get the hyperlinks that are pptx and download them
    # pptx_links = [link[0] for link in hyperlinks if link[0].endswith('.pptx')]
    # for link in pptx_links:
    #     download_file(link, PPTX_FOLDER)

    # convert_all_pptx_in_folder(PPTX_FOLDER, PDF_FOLDER)

    # # get the hyperlinks that are pdf and download them
    pdf_links = [link[0] for link in hyperlinks if link[0].endswith('.pdf')]
    print(f"[getPdfs] Downloading {len(pdf_links)} PDF files...")
    for link in pdf_links:
        download_file(link, PDF_FOLDER)
    print(f"[getPdfs] Finished downloading PDF files")

    # # get the hyperlinks that are webpages and convert them to pdf
    webpage_links = [link[0] for link in hyperlinks if not link[0].endswith('.pdf') and not link[0].endswith('.pptx')]
    print(f"[getPdfs] Converting {len(webpage_links)} webpages to PDF...")
    loop = asyncio.get_event_loop()
    for link in webpage_links:
        local_filename = link.split('/')[-1] + '.pdf'
        pdf_path = os.path.join(PDF_FOLDER, local_filename)
        loop.run_until_complete(convert_webpage_as_pdf(link, pdf_path))
    print(f"[getPdfs] PDF processing complete!")

if __name__ == "__main__":
    main()

