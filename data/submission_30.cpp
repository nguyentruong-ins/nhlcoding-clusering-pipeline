int superstring(string x, string y, int m, int n) {
    // Base case: If either of the strings is empty, return the length of the other string
    if (m == 0) return n;
    if (n == 0) return m;

    // If the last characters match, we have two choices:
    // 1. Include the common character in the super string and recurse on the rest of both strings.
    // 2. Ignore one of the matching characters and recurse on the rest of the strings.
    if (x[m - 1] == y[n - 1]) {
        return 1 + superstring(x, y, m - 1, n - 1);
    } else {
        int choice1 = 1 + superstring(x, y, m - 1, n);  // Ignore the last character of x
        int choice2 = 1 + superstring(x, y, m, n - 1);  // Ignore the last character of y
        return min(choice1, choice2);
    }
}