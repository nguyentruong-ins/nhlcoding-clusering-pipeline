int shortestSuperStringLength(string x, string y, int i, int j) {
	// Base case: If either string is empty, return the length of the other string
	if (i == 0) {
		return j;
	}
	if (j == 0) {
		return i;
	}

	// If the last characters match, reduce both strings and move one step
	if (x[i - 1] == y[j - 1]) {
		return shortestSuperStringLength(x, y, i - 1, j - 1) + 1;
	}

	// If the last characters don't match, try two options and return the minimum
	int option1 = shortestSuperStringLength(x, y, i - 1, j) + 1;
	int option2 = shortestSuperStringLength(x, y, i, j - 1) + 1;

	return min(option1, option2);
}

int superString(string x, string y, int m, int n) {
	return shortestSuperStringLength(x, y, m, n);
}