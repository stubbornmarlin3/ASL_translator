# Main running file
# Aidan Carter
# ASL Interpreter

from dataset import Dataset

def main():
    train = Dataset("./MS-ASL/MSASL_train.json", "./Train")
    valid = Dataset("./MS-ASL/MSASL_val.json", "./Valid")
    test = Dataset("./MS-ASL/MSASL_test.json", "./Test")

    train.downloadVideo(0)
    valid.downloadVideo(3)
    test.downloadVideo(20)

if __name__ == "__main__":
    main()