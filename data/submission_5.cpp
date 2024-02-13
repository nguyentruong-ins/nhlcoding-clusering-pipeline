int superString(const string& x, const string& y, int m, int n) {
    if (m == 0) return n;
    if (n == 0) return m;
    if (x[m - 1] == y[n - 1]) {
        return 1 + superString(x, y, m - 1, n - 1);
    } else {
        return min(superString(x, y, m - 1, n), superString(x, y, m, n - 1)) + 1;
    }
}