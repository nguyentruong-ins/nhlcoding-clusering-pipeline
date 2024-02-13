int shortestSuperstring(const string& x, const string& y, int m, int n) {
    if (m == 0) { // x is empty, so the superstring is y
        return n;
    }
    if (n == 0) { // y is empty, so the superstring is x
        return m;
    }
    if (x[m - 1] == y[n - 1]) { // Last characters of x and y are equal
        return 1 + shortestSuperstring(x, y, m - 1, n - 1); // Consider the common character in the superstring
    } else {
        // Find the shortest superstring by considering each string without the last character
        int superstringWithoutLastX = 1 + shortestSuperstring(x, y, m - 1, n); // Consider x without the last character
        int superstringWithoutLastY = 1 + shortestSuperstring(x, y, m, n - 1); // Consider y without the last character
        return min(superstringWithoutLastX, superstringWithoutLastY); // Return the minimum superstring length
    }
}

int superString(const string& x, const string& y, int m, int n) {
    int shortestLength = shortestSuperstring(x, y, m, n);
    return shortestLength;
}