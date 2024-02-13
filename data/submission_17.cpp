string largestsub(string x, string y){
    int len1 = x.length();
    int len2 = y.length();
    if (len1 == 0 || len2 == 0) return "";
    if(x[len1-1] == y[len2-1]) return largestsub(x.substr(0,len1-1),y.substr(0,len2-1)) + x[len1-1];
    else {
        string s1 = largestsub(x.substr(0,len1-1),y);
        string s2 = largestsub(x,y.substr(0,len2-1));
        if(s1.length()>s2.length()) return s1;
        else return s2;
    }
}

int superString(string x, string y, int m, int n) {
    string s = largestsub(x,y);
    int len = s.length();
    return m+n-len;
}