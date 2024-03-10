import re

def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

program_file = """
string findLargestSubsequence(string str1, string str2) {
    int len1 = str1.length();
    int len2 = str2.length();

    // Base case: Nếu một trong hai chuỗi là rỗng, trả về chuỗi rỗng
    if (len1 == 0 || len2 == 0) {
        return "";
    }

    // Nếu ký tự cuối cùng của cả hai chuỗi giống nhau
    if (str1[len1 - 1] == str2[len2 - 1]) {
        // Gọp ký tự cuối cùng vào subsequence và tiếp tục đệ quy với hai chuỗi bỏ ký tự cuối
        return findLargestSubsequence(str1.substr(0, len1 - 1), str2.substr(0, len2 - 1)) + str1[len1 - 1];
    }
    else {
        // Nếu ký tự cuối của cả hai chuỗi không giống nhau, so sánh hai trường hợp:
        // 1. Xóa ký tự cuối của str1 và tìm subsequence chung
        // 2. Xóa ký tự cuối của str2 và tìm subsequence chung
        string subsequence1 = findLargestSubsequence(str1.substr(0, len1 - 1), str2);
        string subsequence2 = findLargestSubsequence(str1, str2.substr(0, len2 - 1));

        // Trả về subsequence có độ dài lớn hơn giữa hai trường hợp trên
        if (subsequence1.length() > subsequence2.length()) {
            return subsequence1;
        }
        else {
            return subsequence2;
        }
    }
}



int superString(string x, string y, int m, int n) {
    string k = findLargestSubsequence(x,y);
    int f= k.length();
    return m+n-f;
}
"""

print(comment_remover(program_file))