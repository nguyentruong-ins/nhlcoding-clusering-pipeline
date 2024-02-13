int superString(string x, string y, int m, int n) {
      if (m == 0) {
        return n; // If the first string is empty, the superstring length is the length of the second string.
    }
    if (n == 0) {
        return m; // If the second string is empty, the superstring length is the length of the first string.
    }
    
    // If the last characters of both strings match, we can merge them and reduce the problem to the rest of the strings.
    if (x[m - 1] == y[n - 1]) {
        return superString(x, y, m - 1, n - 1) + 1;
    } else {
        // If the last characters don't match, we have two options: either append one character from x or one character from y.
        int option1 = superString(x, y, m - 1, n) + 1;
        int option2 = superString(x, y, m, n - 1) + 1;
        return min(option1, option2);
    }
}