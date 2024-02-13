int longestSubSeq(string x , string y , int m , int n){
    if(m==0 || n==0) return 0;
    if(x[m-1]==y[n-1]) return 1 + longestSubSeq(x,y,m-1,n-1);
    else return max(longestSubSeq(x,y,m-1,n),longestSubSeq(x,y,m,n-1));
}

int superString(string x, string y, int m, int n) {
    return m+n-longestSubSeq(x,y,m,n);
}