int shortestSuperString(string x, string y, int m, int n){
    if(m==0) return n;
    if(n==0) return m;
    
    if(x[m-1] == y[n-1])
        return 1+shortestSuperString(x, y, m-1, n-1);
    else
        return 1 + min(shortestSuperString(x,y,m-1,n), shortestSuperString(x,y,m,n-1));
}

int superString(string x, string y, int m, int n) {
    return shortestSuperString(x, y, m, n);
}