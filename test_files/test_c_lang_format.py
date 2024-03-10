import subprocess

def format_code(code, style='google'):
    # Run clang-format with the specified style and capture the formatted code
    formatted_code = subprocess.run(
        ['clang-format', '-style=' + style], 
        input=code.encode('utf-8'), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    # Return the formatted code as a string
    return formatted_code.stdout.decode('utf-8')

# Example usage
if __name__ == "__main__":
    # Example C++ code
    cpp_code = """
    int superString(string x, string y, int m, int n) {
    if(m==0) return n;
    if(n==0) return m;
    if(x[m-1]==y[n-1])
    return 1+superString(x,y,m-1,n-1);
    return 1+ min(superString(x,y,m-1,n),superString(x,y,m,n-1));
}
    """

    # Format the code using clang-format
    formatted_cpp_code = format_code(cpp_code)

    # Print the formatted code
    print(formatted_cpp_code)
