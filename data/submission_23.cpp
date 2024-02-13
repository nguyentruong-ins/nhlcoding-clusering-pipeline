int max(int x, int y) {
    return (x>y) ? x : y;
}

int lcs(string X, string Y, int m, int n)
{
    if (m == 0 || n == 0) return 0;
    if (X[m - 1] == Y[n - 1]) return lcs(X, Y, m - 1, n - 1) + 1;
    else return max(lcs(X, Y, m, n - 1), lcs(X, Y, m - 1, n));
}

int superString(string X, string Y, int m, int n)
{
    int lcs_length = lcs(X, Y, m, n);
    return m + n - lcs_length;
}