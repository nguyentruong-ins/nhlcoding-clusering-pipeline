int superString(string x, string y, int m, int n) {
    if(m == 0) return n;
    if(n == 0) return m;
    if(x[0] == y[0]) return 1 + superString(x.substr(1), y.substr(1), m - 1, n - 1);
    else {
        int temp1 = superString(x.substr(1), y, m - 1, n);
        int temp2 = superString(x, y.substr(1), m, n - 1);
        if(temp1 > temp2) return 1 + temp2;
        else return 1 + temp1;
    }
}