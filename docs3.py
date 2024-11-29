import os
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image

class PDFExtractor:
    def __init__(self, pdf_path='./[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf'):
        """
        PDF 데이터 추출기 초기화
        
        Args:
            pdf_path (str): PDF 파일 경로
        """
        self.pdf_path = pdf_path
        # 출력 디렉토리 생성 (없다면)
        os.makedirs('output', exist_ok=True)

    def extract_text(self):
        """
        PDF에서 텍스트 추출
        
        Returns:
            str: 추출된 텍스트
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # 모든 페이지의 텍스트를 결합
                full_text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
            
            # 텍스트를 파일로 저장
            with open('output/text.txt', 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print("텍스트 추출 완료: output/text.txt")
            return full_text
        
        except Exception as e:
            print(f"텍스트 추출 중 오류 발생: {e}")
            return None

    def extract_tables(self):
        """
        PDF에서 테이블 추출 및 Excel 파일로 저장
        
        Returns:
            list: 추출된 데이터프레임 목록
        """
        tables = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 페이지에서 테이블 추출
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        # 첫 번째 행을 헤더로 사용
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)
                        
                        # 각 테이블을 개별 시트로 저장
                        with pd.ExcelWriter(f'output/excel_{page_num+1}_{table_num+1}.xlsx') as writer:
                            df.to_excel(writer, index=False)
            
            print(f"{len(tables)}개의 테이블 추출 완료")
            return tables
        
        except Exception as e:
            print(f"테이블 추출 중 오류 발생: {e}")
            return []

    def extract_images(self):
        """
        PDF에서 이미지 추출
        
        Returns:
            list: 추출된 이미지 경로 목록
        """
        extracted_images = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 페이지의 이미지 추출
                    images = page.images
                    
                    for img_num, img in enumerate(images):
                        try:
                            # 이미지 자르기
                            x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                            cropped_page = page.crop((x0, top, x1, bottom))
                            
                            # 이미지 변환 및 저장
                            image = cropped_page.to_image()
                            image_path = f'output/img_{page_num+1}_{img_num+1}.png'
                            image.save(image_path)
                            
                            extracted_images.append(image_path)
                        
                        except Exception as img_error:
                            print(f"이미지 추출 중 오류 발생: {img_error}")
            
            print(f"{len(extracted_images)}개의 이미지 추출 완료")
            return extracted_images
        
        except Exception as e:
            print(f"이미지 추출 중 오류 발생: {e}")
            return []

def main():
    # PDF 추출기 초기화
    extractor = PDFExtractor('./[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf')
    
    # 텍스트 추출
    extractor.extract_text()
    
    # 테이블 추출
    extractor.extract_tables()
    
    # 이미지 추출
    extractor.extract_images()

if __name__ == '__main__':
    main()

# 필요한 종속성 설치 명령어
# pip install pdfplumber pandas pillow pytesseract