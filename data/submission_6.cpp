int lcs(string x, string y, int m, int n) {
    if (m == 0 || n == 0)
        return 0;
    else if (x[m - 1] == y[n - 1])
        return 1 + lcs(x, y, m - 1, n - 1);
    else
        return max(lcs(x, y, m, n - 1), lcs(x, y, m - 1, n));
}

int superString(string x, string y, int m, int n) {
    int len_lcs = lcs(x, y, m, n);
    return (m + n - len_lcs);
}