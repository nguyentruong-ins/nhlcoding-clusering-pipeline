int min(int a, int b) {
    return (a < b) ? a : b;
}

int superStringHelper(const string& x, const string& y, int m, int n) {
    // Base cases
    if (m == 0) return n;
    if (n == 0) return m;

    // If the last characters are the same, consider only one instance of the common character
    if (x[m - 1] == y[n - 1]) {
        return 1 + superStringHelper(x, y, m - 1, n - 1);
    }

    // Otherwise, consider both strings without the last character and choose the minimum
    return 1 + min(superStringHelper(x, y, m - 1, n), superStringHelper(x, y, m, n - 1));
}

int superString(const string& x, const string& y, int m, int n) {
    // Call the helper function
    return superStringHelper(x, y, m, n);
}