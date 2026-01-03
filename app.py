import cv2
import numpy as np
from pdf2image import convert_from_path
import PyPDF2
import re
import json
import os
from pathlib import Path


class CrosswordExtractor:
    def __init__(self, pdf_path, grid_size=15, dpi=300):
        """
        Initialize the crossword extractor.

        Args:
            pdf_path: Path to the PDF file
            grid_size: Size of the crossword grid (default 15x15)
            dpi: Resolution for PDF conversion
        """
        self.pdf_path = pdf_path
        self.grid_size = grid_size
        self.dpi = dpi
        self.grid = None
        self.cell_size = None

    def extract_image_from_pdf(self, page_num=0):
        """Convert PDF page to image."""
        pages = convert_from_path(
            self.pdf_path, dpi=self.dpi, first_page=page_num+1, last_page=page_num+1)
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

    def detect_grid(self, image):
        """Detect the crossword grid in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour (likely the grid)
        grid_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:  # Rectangle
                    max_area = area
                    grid_contour = approx

        if grid_contour is None:
            raise ValueError("Could not detect crossword grid")

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(grid_contour)

        return image[y:y+h, x:x+w], (x, y, w, h)

    def extract_grid_cells(self, grid_image):
        """Extract individual cells and determine which are filled (grey/black)."""
        h, w = grid_image.shape[:2]
        cell_w = w // self.grid_size
        cell_h = h // self.grid_size
        self.cell_size = (cell_w, cell_h)

        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w

                # Sample the center of the cell to avoid edge artifacts
                y_center = (y1 + y2) // 2
                x_center = (x1 + x2) // 2
                sample_size = min(cell_h, cell_w) // 3

                cell_sample = gray[
                    y_center - sample_size:y_center + sample_size,
                    x_center - sample_size:x_center + sample_size
                ]

                avg_brightness = np.mean(cell_sample)

                # If cell is dark (grey/black), mark as filled (0)
                # If cell is light (white), mark as empty (1)
                grid[row, col] = 1 if avg_brightness > 180 else 0

        self.grid = grid
        return grid

    def extract_text_from_pdf(self, page_num=0):
        """Extract text content from PDF to get clue numbers."""
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            page = reader.pages[page_num]
            text = page.extract_text()
        return text

    def find_clue_starts_and_lengths(self):
        """Identify starting positions and lengths of across and down clues."""
        rows, cols = self.grid_size, self.grid_size

        across_clues = {}
        down_clues = {}

        # Auto-number cells that start words
        clue_num = 1
        for row in range(rows):
            for col in range(cols):
                if self.grid[row, col] == 1:  # White cell
                    is_across_start = False
                    is_down_start = False

                    # Check if this is the start of an across word
                    if (col == 0 or self.grid[row, col-1] == 0) and col + 1 < cols and self.grid[row, col+1] == 1:
                        is_across_start = True

                    # Check if this is the start of a down word
                    if (row == 0 or self.grid[row-1, col] == 0) and row + 1 < rows and self.grid[row+1, col] == 1:
                        is_down_start = True

                    if is_across_start or is_down_start:
                        if is_across_start:
                            # Calculate length
                            length = 1
                            for c in range(col + 1, cols):
                                if self.grid[row, c] == 1:
                                    length += 1
                                else:
                                    break
                            across_clues[clue_num] = {
                                'position': (row, col),
                                'length': length
                            }

                        if is_down_start:
                            # Calculate length
                            length = 1
                            for r in range(row + 1, rows):
                                if self.grid[r, col] == 1:
                                    length += 1
                                else:
                                    break
                            down_clues[clue_num] = {
                                'position': (row, col),
                                'length': length
                            }

                        clue_num += 1

        return across_clues, down_clues

    def process(self):
        """Main processing pipeline."""
        print("Converting PDF to image...")
        image = self.extract_image_from_pdf()

        print("Detecting grid...")
        grid_image, bounds = self.detect_grid(image)

        print(
            f"Extracting cells from {self.grid_size}x{self.grid_size} grid...")
        self.extract_grid_cells(grid_image)

        print("Finding clues...")
        across, down = self.find_clue_starts_and_lengths()

        return {
            'across': across,
            'down': down,
            'grid_size': (self.grid_size, self.grid_size),
            'grid': self.grid.tolist()
        }


# Example usage
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Get all PDF files from input directory
    input_dir = Path("input")
    if not input_dir.exists():
        print("Error: 'input' folder not found. Please create it and add PDF files.")
        exit(1)

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in 'input' folder.")
        exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) to process.\n")

    # Process each PDF
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print('='*60)

        try:
            # Initialize extractor with known grid size
            extractor = CrosswordExtractor(str(pdf_path), grid_size=15)

            # Process the crossword
            result = extractor.process()

            # Print results to console
            print("\nACROSS CLUES:")
            print("-" * 50)
            for num in sorted(result['across'].keys()):
                clue = result['across'][num]
                print(
                    f"{num}. Position: (row={clue['position'][0]}, col={clue['position'][1]}), Length: {clue['length']}")

            print("\nDOWN CLUES:")
            print("-" * 50)
            for num in sorted(result['down'].keys()):
                clue = result['down'][num]
                print(
                    f"{num}. Position: (row={clue['position'][0]}, col={clue['position'][1]}), Length: {clue['length']}")

            print(
                f"\nGrid size: {result['grid_size'][0]} x {result['grid_size'][1]}")

            # Optional: visualize the grid
            print("\nGrid visualization (■=white, □=black):")
            for row in result['grid']:
                print(''.join(['■' if cell else '□' for cell in row]))

            # Save to JSON
            output_filename = pdf_path.stem + ".json"
            output_path = output_dir / output_filename

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Processing complete! Check the 'output' folder for JSON files.")
    print('='*60)
