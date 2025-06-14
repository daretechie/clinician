import docx
import sys

def get_text(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(get_text(sys.argv[1]))
    else:
        print("Please provide a filename")
