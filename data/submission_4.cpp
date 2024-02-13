int superString(const string& x, const string& y, int m, int n) {
    // Base cases
    if (m == 0) return n;
    if (n == 0) return m;

    // If the last characters of x and y are the same, ignore them and recursively find the super string
    if (x[m - 1] == y[n - 1])
        return 1 + superString(x, y, m - 1, n - 1);

    // If the last characters of x and y are different, consider both possibilities:
    // 1. Ignore the last character of x and find the super string of the remaining x and y
    // 2. Ignore the last character of y and find the super string of x and the remaining y
    return 1 + min(superString(x, y, m - 1, n), superString(x, y, m, n - 1));
}