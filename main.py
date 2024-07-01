from pdf2image import convert_from_path
import cv2

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.empty_cache()
import re
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
from paddleocr import PaddleOCR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import json
import shutil
import logging
import sys
import paddle
import pytesseract



paddle.disable_signal_handler()
PaddleOCR(show_log=False)

tf.get_logger().setLevel(logging.ERROR)
#from spac import process_with_spacy

# Suppress Tesseract logging
pytesseract.pytesseract.tesseract_cmd = r'<path_to_your_tesseract_executable>'
os.environ['OMP_THREAD_LIMIT'] = '1'

class Utilities:
    class SuppressOutput:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        
        def __exit__(self, exc_type, exc_value, traceback):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    @staticmethod
    def suppress_output():
        return Utilities.SuppressOutput()

class TableLinesRemover:
    def __init__(self, image):
        self.original_image = image.copy()
        self.image = image
        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def execute(self):
        self.grayscale_image()
        self.threshold_image()
        self.invert_image()
        self.detect_lines()
        self.enhance_lines()
        self.merge_lines_with_original()
        return self.merged_image

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grey, 127, 255, cv2.THRESH_BINARY)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def detect_lines(self):
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        self.detect_horizontal = cv2.morphologyEx(self.inverted_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        self.detect_vertical = cv2.morphologyEx(self.inverted_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    def enhance_lines(self):
        # Combine horizontal and vertical lines
        self.lines = cv2.add(self.detect_horizontal, self.detect_vertical)

        # Enhance the lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.enhanced_lines = cv2.dilate(self.lines, kernel, iterations=2)

    def merge_lines_with_original(self):
        # Convert enhanced lines to 3-channel
        enhanced_lines_color = cv2.cvtColor(self.enhanced_lines, cv2.COLOR_GRAY2BGR)

        # Create a mask of the lines
        _, mask = cv2.threshold(self.enhanced_lines, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Convert masks to 3-channel
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        # Blend the lines with the original image
        blended_lines = cv2.addWeighted(self.original_image, 1, enhanced_lines_color, 0.5, 0)

        # Use the mask to only keep the blended lines where lines were detected
        self.merged_image = cv2.bitwise_and(blended_lines, mask_3ch) + cv2.bitwise_and(self.original_image, mask_inv_3ch)

        # Optional: Sharpen the result to make text clearer
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.merged_image = cv2.filter2D(self.merged_image, -1, kernel)
    def store_process_image(self, file_path, image):
        cv2.imwrite(file_path, image)

class BaseFunctions:
    def __init__(self):
        self.utils = Utilities()
        self.x = None  

    def extract_part_code(self, description):
        parts = re.split(r'[^a-zA-Z0-9]+', description.replace(' ', ''))
        for part in parts:
            if len(part) >= 13 and part[:2] in ['UF', 'TL', 'FL', 'CD', 'CC']:
                return part[:13]
        return ""

    def split_number_and_letters(self, text):
        pattern = re.compile(r'(\d+\.\d+|\d+)(?:\s*)([a-zA-Z]+)')
        matches = re.findall(pattern, text)
        if matches:
            return matches
        else:
            return None

    def parse_hsn_string(self, product_str):
        code_pattern = re.compile(r'\b\d+\b') 
        code_matches = code_pattern.findall(product_str)
        if code_matches:
            code = code_matches[0]
            description = product_str.replace(code, '', 1).strip()
            return code, description
        else:
            return None, product_str.strip()

    def split_numeric_and_words(self, input_string):
        numeric_part = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', input_string)
        word_part = re.findall(r'[a-zA-Z]+', input_string)
        return numeric_part, word_part

    def enhance_si_no_column(self, image, si_no_col_width):
        si_no_col = image[:, :si_no_col_width]
        si_no_col = cv2.cvtColor(si_no_col, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_si_no_col = clahe.apply(si_no_col)
        if len(image.shape) == 3:
            enhanced_si_no_col = cv2.cvtColor(si_no_col, cv2.COLOR_GRAY2BGR)
        return enhanced_si_no_col

class AMIFunctions(BaseFunctions):
    def __init__(self):
        super().__init__()

    def enhance_image(self, cropped_image_filename):
        if not os.path.exists(cropped_image_filename):
            print(f"File not found: {cropped_image_filename}")
            return None
        
        im = cv2.imread(cropped_image_filename)
        if im is None:
            print("Failed to read the image. Please check the file path and integrity.")
            return None
        
        si_no_col_width = 48  # Adjust this value based on your image
        enhanced_si_no_col = self.enhance_si_no_column(im, si_no_col_width)
        mask = np.zeros(im.shape[:2], dtype=np.uint8)
        mask[:, si_no_col_width:] = 1

        negative_image = 255 - im
        gray_negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        enhanced_negative_image = gray_negative_image.copy()
        enhanced_masked_area = clahe.apply(gray_negative_image[mask == 1].reshape(-1, gray_negative_image.shape[1] - si_no_col_width))
        enhanced_negative_image[mask == 1] = enhanced_masked_area.flatten()

        final_image = im.copy()
        final_image[:, si_no_col_width:] = cv2.cvtColor(enhanced_negative_image[:, si_no_col_width:], cv2.COLOR_GRAY2BGR)
        final_image[:, :si_no_col_width] = enhanced_si_no_col
        output_filename = os.path.join("cropped", f"enhanced_page_1.png")
        cv2.imwrite(output_filename, final_image)
        
        return final_image

    def merge_data(self, array):
        merged_data = []
        headers = array[0]
        current_item = {}
        skip_next_empty = False
        for row in array[2:]:
            si_no = row[0]
            if "SGST" in row or "CGST" in row or "Total" in row:  
                break
            if row[1] == '' and row[2] == '' and row[3] == '' and row[4] == '' and row[5] == '' and row[6] == '' :
                continue
            if skip_next_empty:
                skip_next_empty = False
                continue
            if si_no:  
                if current_item:
                    merged_data.append(current_item)  
                current_item = {headers[0]: si_no}  
            for i in range(1, len(headers)):
                if row[i]:  
                    if headers[i] in current_item: 
                        current_item[headers[i]] += " " + row[i].strip()  
                    else:
                        current_item[headers[i]] = row[i].strip()  
            if not any(row[1:]):  
                skip_next_empty = True
            
        if current_item:  
            merged_data.append(current_item)
        return merged_data

    def rename_keys(self, item):
        renamed_item = {}
        for key, value in item.items():
            if "description" in key.lower() or "desc" in key.lower():
                new_key = "description"
            elif "hsn" in key.lower():
                new_key = "hsn"
                value = value.strip()
                if not value.isdigit():
                    value = str(value[:8])
                renamed_item[new_key] = value
            elif "quantity" in key.lower():
                new_key = "quantity"
            elif "rate" in key.lower():
                new_key = "rate"
            elif "amount" in key.lower() or "value" in key.lower():
                new_key = "amount"
            elif "part" in key.lower():
                new_key = "partCode"
            elif "unit" in key.lower() or "uom" in key.lower():
                new_key = "uom"
            else:
                new_key = key.lower()
            renamed_item[new_key] = value

        if "description" in renamed_item:
            part_code = self.extract_part_code(renamed_item["description"])
            if len(part_code) == 13 and part_code != "":
                renamed_item["partCode"] = part_code

        if "rate" not in renamed_item:
            for key1, value1 in item.items():
                if "per" in key1.lower():
                    match = re.compile(r"(\d+\.?\d*)\s*(.*)").match(value1)
                    if match:
                        renamed_item["rate"] = match.group(1)
                        renamed_item["per"] = match.group(2)
                        break  
                    else:
                        renamed_item["rate"] = value
        for key, value in item.items():
            if "quantity" in key.lower():
                if value.isdigit():  
                    renamed_item["quantity"] = value
                    renamed_item["uom"] = ""  
                else:
                    numeric_part, word_part = self.split_numeric_and_words(value)
                    if numeric_part:
                        if word_part:
                            renamed_item["quantity"] = numeric_part[0]
                            renamed_item["uom"] = word_part[0]
                        else:
                            renamed_item["quantity"] = numeric_part[0]
                            renamed_item["uom"] = "" 
                    else:
                        print(f"Invalid quantity provided: {value}")  
        return renamed_item


class MPLFunctions(BaseFunctions):
    def __init__(self):
        super().__init__()

    def enhance_image(self, cropped_image_filename):
        if not os.path.exists(cropped_image_filename):
            print(f"File not found: {cropped_image_filename}")
            return None
        
        im = cv2.imread(cropped_image_filename)
        lineEnchancing = TableLinesRemover(im)
        im = lineEnchancing.execute()
        if im is None:
            print("Failed to read the image. Please check the file path and integrity.")
            return None
        
        si_no_col_width = 2  # Adjust this value based on your image
        if self.x is not None:
            si_no_col_width = self.x
        enhanced_si_no_col = self.enhance_si_no_column(im, si_no_col_width)
        mask = np.zeros(im.shape[:2], dtype=np.uint8)
        mask[:, si_no_col_width:] = 1

        negative_image = 255 - im
        gray_negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
        enhanced_negative_image = gray_negative_image.copy()
        enhanced_masked_area = clahe.apply(gray_negative_image[mask == 1].reshape(-1, gray_negative_image.shape[1] - si_no_col_width))
        enhanced_negative_image[mask == 1] = enhanced_masked_area.flatten()

        final_image = im.copy()
        final_image[:, si_no_col_width:] = cv2.cvtColor(enhanced_negative_image[:, si_no_col_width:], cv2.COLOR_GRAY2BGR)
        final_image[:, :si_no_col_width] = enhanced_si_no_col
        
        output_filename = os.path.join("cropped", f"enhanced_page_1.png")
        cv2.imwrite(output_filename, final_image)
        
        return final_image

    def merge_data(self, array):
        headers = [''] * len(array[0])
        for row in array[0:3]:
            for i, data in enumerate(row):
                if(data != ""):
                    headers[i] +=  data
        merged_data = []
        #headers = array[0]
        current_item = {}
        skip_next_empty = False
        for row in array[3:]:
            si_no = row[0]
            if "SGST" in row or "CGST" in row or "Total" in row:  
                break
            if row[1] == '' and row[2] == '' and row[3] == '' and row[4] == '' and row[5] == '' and row[6] == '' :
                continue
            if skip_next_empty:
                skip_next_empty = False
                continue
            if si_no:  
                if current_item:
                    merged_data.append(current_item)  
                current_item = {headers[0]: si_no}  
            for i in range(1, len(headers)):
                if row[i]:  
                    if headers[i] in current_item: 
                        current_item[headers[i]] += " " + row[i].strip()  
                    else:
                        current_item[headers[i]] = row[i].strip()  
            if not any(row[1:]):  
                skip_next_empty = True
            
        if current_item:  
            merged_data.append(current_item)
        return merged_data

    def rename_keys(self, item):
        renamed_item = {}
        for key, value in item.items():
            if "description" in key.lower() or "desc" in key.lower():
                new_key = "description"
            elif "customer" in key.lower() or "material" in key.lower():
                new_key = "partCode"
                if(len(value)>13):
                    arr = value.split()
                    value = arr[0] + arr[-1]
            elif "hsn" in key.lower():
                new_key = "hsn"
                value = value.strip()
                if not value.isdigit():
                    value = str(value[-8:-1])
                renamed_item[new_key] = value
            elif "qty" in key.lower() and "unit" in key.lower() and "&" in key.lower():
                new_key = "quantity"
            elif "rate" in key.lower() and "amount" in key.lower() and "&" in key.lower():
                new_key = "rate & amount"
            elif "rate" in key.lower() and "amount" in key.lower() and "sgst" in key.lower():
                new_key = "sgst"
            elif "rate" in key.lower() and "amount" in key.lower() and "cgst" in key.lower():
                new_key = "cgst"
            else:
                new_key = key.lower()
            renamed_item[new_key] = value

        if "description" in renamed_item:
            part_code = self.extract_part_code(renamed_item["description"])
            if len(part_code) == 13 and part_code != "":
                renamed_item["partCode"] = part_code
        if "quantity" in renamed_item:
            quantity_info = renamed_item["quantity"]
            if " " in quantity_info:
                quantity, unit = quantity_info.split(maxsplit=1)
                renamed_item["quantity"] = quantity
                renamed_item["uom"] = unit
        if "rate & amount" in renamed_item:
            rate_amount_info = renamed_item["rate & amount"]
            if " " in rate_amount_info:
                rate, amount = rate_amount_info.split(maxsplit=1)
                renamed_item["rate"] = rate
                renamed_item["amount"] = amount
                del renamed_item["rate & amount"]
        if "sgst" in renamed_item:
            rate_amount_info = renamed_item["sgst"]
            if " " in rate_amount_info:
                rate, amount = rate_amount_info.split(maxsplit=1)
                renamed_item["sgst rate"] = rate
                renamed_item["sgst amount"] = amount
                del renamed_item["sgst"]
        if "cgst" in renamed_item:
            rate_amount_info = renamed_item["cgst"]
            if " " in rate_amount_info:
                rate, amount = rate_amount_info.split(maxsplit=1)
                renamed_item["cgst rate"] = rate
                renamed_item["cgst amount"] = amount
                del renamed_item["cgst"]
        
        return renamed_item

class PDFProcessor:
    def __init__(self):
        directories = ["pages", "cropped", "content"]
        result_file = "result.json"
        self.x = None
        if os.path.exists(result_file):
            os.remove(result_file)

        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            
            os.makedirs(directory)
        self.utils = Utilities()
        self.ami_functions = AMIFunctions()
        self.mpl_functions = MPLFunctions()
        # Initialize OCR engines, etc.
        with self.utils.suppress_output():
            self.tesseract_ocr = TesseractOCR(n_threads=4, lang="eng")
            self.ocr = PaddleOCR(
                            drop_score=0.4,
                            cls=True,
                            det_db_thresh=0.5,
                            det_model_dir='PaddleOCR/models/en_PP-OCRv3_det_distill_train',
                            rec_model_dir='PaddleOCR/models/en_PP-OCRv3_rec_train',
                            cls_model_dir='PaddleOCR/models/ch_ppocr_mobile_v2.0_cls_slim_train',
                            use_angle_cls=True,
                            lang='en',
                            use_gpu=True,
                            use_mp=True,
                            total_process_num=4,
                            use_tensorrt=True,
                            rec_algorithm='CRNN',
                            rec_batch_num=1,
                            rec_image_shape="3,  32,  480"
                        )

    def process_pdf(self, pdf_path, pdf_name, document_type):
        # Main processing logic
        with self.utils.suppress_output():
            images = convert_from_path(f"{pdf_path}/{pdf_name}")
        results_list = []
        for i, image in enumerate(images):
            image_name = f"{pdf_name}_page_{i + 1}"
            image_path = os.path.join("pages", f"{image_name}.png")
            image.save(image_path, 'PNG')
            result = self.process_image(image_name, image_path, document_type)
            if result not in results_list:
                results_list.append(result)
        return results_list

    def process_image(self, image_name, image_path, document_type):
        # Image processing logic
        extracted_tables = self.extract_tables_from_image(image_path)
        recy1, recy2 = self.find_relevant_areas(extracted_tables)
        
        if recy1 is not None and recy2 is not None:
            cropped_image_filename = self.crop_image(image_name, image_path, recy1, recy2)
        else:
            return []
        
        if not cropped_image_filename:
            return []
        
        
        if(document_type == "ami" or document_type == "vignesh"):
            functions = self.ami_functions
        elif(document_type == "mpl"):
            functions = self.mpl_functions
        else:
            pass
        functions.x = self.x 
        functions.x = 2
        enhanced_image = functions.enhance_image(cropped_image_filename)
        
        if enhanced_image is None:
            return []
        
        out_array = self.perform_ocr_and_nms(image_name, enhanced_image)
        
        if out_array.size == 0:
            return []
        print(out_array)
        merged_data = functions.merge_data(out_array)
        print(merged_data)
        
        renamed_data = [functions.rename_keys(item) for item in merged_data]
        
        return renamed_data

    def extract_tables_from_image(self, image_path):
        img_from_path = Img2TableImage(src=image_path)
        with self.utils.suppress_output():
            extracted_tables = img_from_path.extract_tables(ocr=self.tesseract_ocr)
        return extracted_tables


    def find_relevant_areas(self, extracted_tables):
        recy1, recy2, search_check, total_check, search_desc = None, None, 0, 0, 0
        self.x = None
        for table in extracted_tables:
            for row in table.content.values():
                for cell in row:
                    try:
                        if recy1 is None and re.search(r"(hsn|description|amount|quantity)", cell.value.lower()) and search_check == 0:
                            recy1 = cell.bbox.y1
                            search_check += 1
                    except:
                        pass
                    try:
                        if self.x is None and re.search(r"desc", cell.value.lower()) and search_desc == 0:
                            self.x = cell.bbox.x1 
                            print(self.x)
                            search_desc += 1
                    except:
                        pass
                    try:
                        if recy2 is None and re.search(r"(total)", cell.value.lower()) and total_check == 0:
                            recy2 = cell.bbox.y2
                            total_check += 1
                    except:
                        pass
        return recy1, recy2

    def crop_image(self, image_name, image_path, recy1, recy2):
        table_img = cv2.imread(image_path)
        cropped_image = table_img[recy1:recy2, :]
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        left_crop_index = next((j for j in range(edges.shape[1]) if np.any(edges[:, j])), None)
        right_crop_index = next((j for j in range(edges.shape[1] - 1, -1, -1) if np.any(edges[:, j])), None)
        
        if left_crop_index is not None and right_crop_index is not None:
            cropped_img = cropped_image[:, left_crop_index-6:right_crop_index+2]
            cropped_image_filename = os.path.join("cropped", f"{image_name}_cropped_page_1.png")
            cv2.imwrite(cropped_image_filename, cropped_img)
            return cropped_image_filename
        return None


    def perform_ocr_and_nms(self, image_name, enhanced_image):
        output = self.ocr.ocr(enhanced_image)[0]
        
        boxes = [line[0] for line in output]
        texts = [line[1][0] for line in output]
        probabilities = [line[1][1] for line in output]

        horiz_boxes, vert_boxes = [], []
        for box in boxes:
            x_h, x_v = 0, int(box[0][0])
            y_h, y_v = int(box[0][1]), 0
            width_h, width_v = enhanced_image.shape[1], int(box[2][0] - box[0][0])
            height_h, height_v = int(box[2][1] - box[0][1]), enhanced_image.shape[0]

            horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
            vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])
        
        horiz_out = tf.image.non_max_suppression(
            horiz_boxes, probabilities, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')
        )
        vert_out = tf.image.non_max_suppression(
            vert_boxes, probabilities, max_output_size=1000, iou_threshold=0.1, score_threshold=float('-inf')
        )

        horiz_lines = np.sort(np.array(horiz_out))
        vert_lines = np.sort(np.array(vert_out))
        
        self.save_detection_images(image_name, enhanced_image, boxes, texts)
        self.save_horiz_vert_images(image_name, enhanced_image, horiz_boxes, vert_boxes, horiz_lines, vert_lines)
        self.save_nms_images(image_name, enhanced_image, horiz_boxes, vert_boxes, horiz_lines, vert_lines)
        
        out_array = self.construct_output_array(horiz_lines, vert_lines, horiz_boxes, vert_boxes, boxes, texts)
        
        return out_array

    def save_detection_images(self, image_name, enhanced_image, boxes, texts):
        image_boxes = enhanced_image.copy()
        for box, text in zip(boxes, texts):
            cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1)
            cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0), 1)
        cv2.imwrite(f'content/{image_name}_detections.png', image_boxes)


    def save_horiz_vert_images(self, image_name, enhanced_image, horiz_boxes, vert_boxes, horiz_lines, vert_lines):
        im = enhanced_image.copy()
        for box in horiz_boxes:
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        for box in vert_boxes:
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.imwrite(f'content/{image_name}_horiz_vert.png', im)


    def save_nms_images(self, image_name, enhanced_image, horiz_boxes, vert_boxes, horiz_lines, vert_lines):
        im_nms = enhanced_image.copy()
        for val in horiz_lines:
            cv2.rectangle(im_nms, (horiz_boxes[val][0], horiz_boxes[val][1]), (horiz_boxes[val][2], horiz_boxes[val][3]), (0, 0, 255), 1)
        for val in vert_lines:
            cv2.rectangle(im_nms, (vert_boxes[val][0], vert_boxes[val][1]), (vert_boxes[val][2], vert_boxes[val][3]), (255, 0, 0), 1)
        cv2.imwrite(f'content/{image_name}_im_nms.png', im_nms)

    def construct_output_array(self, horiz_lines, vert_lines, horiz_boxes, vert_boxes, boxes, texts):
        out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]
        unordered_boxes = [vert_boxes[i][0] for i in vert_lines]
        ordered_boxes = np.argsort(unordered_boxes)
        
        def intersection(box_1, box_2):
            return [box_2[0], box_1[1], box_2[2], box_1[3]]

        def iou(box_1, box_2):
            x_1, y_1 = max(box_1[0], box_2[0]), max(box_1[1], box_2[1])
            x_2, y_2 = min(box_1[2], box_2[2]), min(box_1[3], box_2[3])
            inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1, 0)))
            if inter == 0:
                return 0
            box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
            box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
            return inter / float(box_1_area + box_2_area - inter)

        for i in range(len(horiz_lines)):
            for j in range(len(vert_lines)):
                resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])
                for b in range(len(boxes)):
                    the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                    if iou(resultant, the_box) > 0.1:
                        out_array[i][j] = texts[b]

        return np.array(out_array)

def main():
    pdf_processor = PDFProcessor()
    results_list = pdf_processor.process_pdf("pdfs", "test14.pdf", "ami")
    final_json_data = json.dumps(results_list, indent=4)
    if final_json_data:
        with open('result.json', 'w') as json_file:
            json_file.write(final_json_data)
    print("Results saved to result.json")

if __name__ == "__main__":
    main()