//暴力求解原根
int G(int s)
{
    int q[1010]={0};
    for(int i=2;i<=s-2;i++) if ((s-1)%i==0) q[++q[0]]=i;
    for (int i=2;;i++)
    {
        bool B=1;
        for (int j=1;j<=q[0]&&B;j++) if (quick(i,q[j],s)==1) B=0;
        if (B) return i;
    }
    return -1;
}

//非求逆元的BM板子 (未验证)
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define rep(i,a,n) for(int i=a;i<n;i++)
namespace linear
{
    ll mo=1000000009;
    vector<ll> v;
    double a[105][105],del;
    int k;
    struct matrix
    {
        int n;
        ll a[50][50];
        matrix operator * (const matrix & b)const
        {
            matrix c;
            c.n=n;
            rep(i,0,n)rep(j,0,n)c.a[i][j]=0;
            rep(i,0,n)rep(j,0,n)rep(k,0,n)
            c.a[i][j]=(c.a[i][j]+a[i][k]*b.a[k][j]%mo)%mo;
            return c;
        }
    }A;
    bool solve(int n)
    {
        rep(i,1,n+1)
        {
            int t=i;
            rep(j,i+1,n+1)if(fabs(a[j][i])>fabs(a[t][i]))t=j;
            if(fabs(del=a[t][i])<1e-6)return false;
            rep(j,i,n+2)swap(a[i][j],a[t][j]);
            rep(j,i,n+2)a[i][j]/=del;
            rep(t,1,n+1)if(t!=i)
            {
                del=a[t][i];
                rep(j,i,n+2)a[t][j]-=a[i][j]*del;
            }
        }
        return true;
    }
    void build(vector<ll> V)
    {
        v=V;
        int n=(v.size()-1)/2;
        k=n;
        while(1)
        {
            rep(i,0,k+1)
            {
                rep(j,0,k)a[i+1][j+1]=v[n-1+i-j];
                a[i+1][k+1]=1;
                a[i+1][k+2]=v[n+i];
            }
            if(solve(n+1))break;
            n--;k--;
        }
        A.n=k+1;
        rep(i,0,A.n)rep(j,0,A.n)A.a[i][j]=0;
        rep(i,0,A.n)A.a[i][0]=(int)round(a[i+1][A.n+1]);
        rep(i,0,A.n-2)A.a[i][i+1]=1;
        A.a[A.n-1][A.n-1]=1;
    }
    void formula()
    {
        printf("f(n) =");
        rep(i,0,A.n-1)printf(" (%lld)*f(n-%d) +",A.a[i][0],i+1);
        printf(" (%lld)\n",A.a[A.n-1][0]);
    }
    ll cal(ll n)
    {
        if(n<v.size())return v[n];
        n=n-k+1;
        matrix B,T=A;
        B.n=A.n;
        rep(i,0,B.n)rep(j,0,B.n)B.a[i][j]=i==j?1:0;
        while(n)
        {
            if(n&1)B=B*T;
            n>>=1;
            T=T*T;
        }
        ll ans=0;
        rep(i,0,B.n-1)ans=(ans+v[B.n-2-i]*B.a[i][0]%mo)%mo;
        ans=(ans+B.a[B.n-1][0])%mo;
        while(ans<0)ans+=mo;
        return ans;
    }
}

int main()
{
//  vector<ll> V={1 ,4 ,9 ,16,25,36,49};
//  vector<ll> V={1 ,1 ,2 ,3 ,5 ,8 ,13};
//  vector<ll> V={2 ,2 ,3 ,4 ,6 ,9 ,14};
    vector<ll> V={1,1,1,3,5,9,17};//<-----
    linear::build(V);
    linear::formula();
    ll n;
    while(~scanf("%lld",&n))
    {
        printf("%lld\n",linear::cal(n-1));
    }
    return 0;
}

//母函数  这类母函数通常最后转化为多项式乘法 可以用FFT优化
#include<bits/stdc++.h>
using namespace std;
#define maxn 305
int dp[maxn];
const int a[30]={1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289};

int main()
{
    int n;

    for(int i=0;i<17;i++)
    {
        dp[a[i]]++;
        for(int j=1;j+a[i]<maxn&&j<maxn;j++)
        {
            dp[j+a[i]]+=dp[j];
        }
    }
    while(cin>>n)
    {
        if(!n)
            break;
        cout<<dp[n]<<endl;
    }
    return 0;
}

#include<bits/stdc++.h>
using namespace std;
int a[125];
int b[125];
int main()
{
    ios::sync_with_stdio(false);
    int n;
    while(cin>>n)
    {
        memset(a,0,sizeof(a));
        memset(b,0,sizeof(b));
        a[0]=1;
        for(int i=1;i<=n;i++)     //i的上界为有多少个多项式相乘
        {
            for(int j=0;j<=n;j++)  //j的上界为数组上界，即目前的多项式有多少项
            {
                for(int k=0;k+j<=n;k+=i) //k+j<=n的意思是两多项式相乘，只用考虑上界以下的项
                {                        //k每次加i代表第i个多项式每相邻两项的次数差
                    b[j+k]+=a[j];
                   // cout<<j+k<<' '<<b[j+k]<<endl;
                }
            }
            for(int j=0;j<=n;j++)
                a[j]=b[j];
            memset(b,0,sizeof(b));
        }
        cout<<a[n]<<endl;
    }
    return 0;
}





//指数型母函数
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>
 
using namespace std;
 
typedef long long ll;
 
const int maxn = 1e5 + 10;
#define mem(a) memset(a, 0, sizeof a)
 
double a[maxn],b[maxn]; // 注意为浮点型
 
int s1[maxn];
 
double f[11];
void init() {
    mem(a);
    mem(b);
    mem(s1);
    f[0] = 1;
    for (int i = 1; i <= 10; i++) {
        f[i] = f[i - 1] * i;
    }
}
 
int main() {
    int n,m;
    while (~scanf("%d%d", &n, &m)) {
       init();
       for (int i = 0; i < n; i++) {
            scanf("%d", &s1[i]);
       }
        for (int i = 0; i <= s1[0]; i++) a[i] = 1.0 / f[i];
        for (int i = 1; i < n; i++) {
            mem(b);
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= s1[i] && k + j <= m; k++) {
                    b[j + k] += a[j] * 1.0 / f[k]; //注意这里
                }
            }
            memcpy(a, b, sizeof b);
        }
       printf("%.0f\n", a[m] * f[m]);
    }
    return 0;
}




//指数型母函数模板2
//HDU 1521
//给定n个物品数量，取r个物品，求排列方法数
//ans为(1+x/1!+x^2/2!+...+x^num[i]/num[i]!)乘积后的第r项的系数乘r!
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
typedef long long ll;
using namespace std;
const int N = 11;
double c1[N],c2[N];     //注意类型
ll fac[N];
//预处理阶乘
void cal()
{
    fac[0]=1;           //0!会被用到
    for(int i=1; i<N; i++)
        fac[i]=i*fac[i-1];
}

int main()
{
    cal();
    int n,r;
    while(~scanf("%d%d",&n,&r))
    {
        memset(c1,0,sizeof(c1));
        memset(c2,0,sizeof(c2));

        c1[0]=1;
        int num;
        //计算多项式的乘积
        for(int i=1; i<=n; i++)
        {
            scanf("%d",&num);
            if(num==0) continue;
            for(int j=0; j<=r; j++)
            {
                for(int k=0; k<=num&&k+j<=r; k++)
                {
                    c2[k+j]+=c1[j]/fac[k];
                }
            }
            for(int j=0; j<=r; j++)
            {
                c1[j]=c2[j];
                c2[j]=0;
            }
        }

        printf("%lld\n",(ll)(c1[r]*fac[r]+0.5));
    }

    return 0;
}

//杜教筛求phi和mu的前缀和 O(n^(2/3)) 洛谷P4213 注意long long乘法次数过多会被卡常

#include<bits/stdc++.h>
using namespace std;
#define ll long long
const int maxn = 5e6 + 10;
struct djs
{
    bool vis[maxn];
    int prime[maxn],mu[maxn];
    ll phi[maxn];
    unordered_map<int, int> mm;
    unordered_map<int, ll> pp;
    void init(int n)
    {
        memset(vis,0,sizeof(vis));
        int cnt=0;
        mu[1]=phi[1] = 1;
        for (int i=2; i<= n; i++)
        {
            if (!vis[i])
                prime[++cnt] = i, mu[i] = -1, phi[i] = i - 1;
            for (int j = 1; j <= cnt && prime[j] * i <= n; j++)
            {
                vis[i * prime[j]] = 1;
                if (i % prime[j] == 0)
                {
                    phi[i * prime[j]] =phi[i] * prime[j];
                    break;
                }
                else
                {
                    mu[i * prime[j]]=-mu[i];
                    phi[i * prime[j]]=phi[i] * (prime[j] - 1);
                }
            }
        }
        for (int i = 1; i <= n; i++)
            mu[i]+=mu[i-1],phi[i]+=phi[i-1];
    }
    djs()
    {
        init(maxn-5);
        mm.clear();
        pp.clear();
    }
    int pre_mu(int n)
    {
        if (n <= maxn - 5) return mu[n];
        if (mm[n]) return mm[n];

        int l = 2, r, ans = 1;
        for(l=2; l<=n; l=r+1)
        {
            r=n/(n/l);
            ans-=(r-l+1)*pre_mu(n/l);
        }
        return mm[n] = ans;
    }

    long long pre_phi(int n)
    {
        if (n <= maxn - 5) return phi[n];
        if (pp[n]) return pp[n];
        int l = 2, r;
        long long ans = 1LL*n*(n+1)/2;
        for(l=2; l<=n; l=r+1)
        {
            r=n/(n/l);
            ans-=(r-l+1)*pre_phi(n/l);
        }
        return pp[n] = ans;
    }
}D;

signed main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        int n;
        scanf("%d", &n);
        printf("%lld %d\n", D.pre_phi(n),D.pre_mu(n));
    }
    return 0;
}

/*
	博弈论算法
*/



//SG函数模板        Hdu1536 凡是可以化简为Impartial Combinatorial Games的抽象模型（有向图移动棋子）
//都可以尝试利用SG函数 复杂度目前还不太清楚，板子是抄的
//如果堆数或每堆的棋子数太大，可以考虑打标猜规律找循环节   如Codeforces1194D
#include<bits/stdc++.h>
#include<ext/rope>
using namespace std;
using namespace __gnu_cxx;
#define ll long long
#define ld long double
#define ull unsigned long long
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-6;
const int maxn=1e5+10;
const int INF=0x3f3f3f3f;
const double e=2.718281828459045;
#define Accepted 0
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}

//注意 S数组要按从小到大排序 SG函数要初始化为-1 对于每个集合只需初始化1边
//不需要每求一个数的SG就初始化一边
int SG[10100],n,m,s[102],k;//k是集合s的大小 S[i]是定义的特殊取法规则的数组
int dfs(int x)//求SG[x]模板
{
    if(SG[x]!=-1) return SG[x];
    bool vis[110];
    memset(vis,0,sizeof(vis));

    for(int i=0; i<k; i++)
    {
        if(x>=s[i])
        {
            dfs(x-s[i]);
            vis[SG[x-s[i]]]=1;
        }
    }
    int e;
    for(int i=0;; i++)
        if(!vis[i])
        {
            e=i;
            break;
        }
    return SG[x]=e;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    while(cin>>k&&k)
    {
        memset(SG,-1,sizeof(SG));
        for(int i=0; i<k; i++)
            cin>>s[i];
        cin>>m;
        for(int i=0; i<m; i++)
        {
            int sum=0;
            cin>>n;
            for(int i=0,a; i<n; i++)
            {
                cin>>a;
                sum^=dfs(a);
            }
            // printf("SG[%d]=%d\n",num,SG[num]);
            if(sum==0) putchar('L');
            else putchar('W');
        }
        putchar('\n');
    }
    return Accepted;
}

//还是SG函数
//神仙网友的板子也挺不错的，贴在这里    
//https://www.cnblogs.com/dyllove98/p/3194312.html

#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;
//注意 S数组要按从小到大排序 SG函数要初始化为-1 对于每个集合只需初始化1边
//不需要每求一个数的SG就初始化一边
int SG[10100],n,m,s[102],k;//k是集合s的大小 S[i]是定义的特殊取法规则的数组
int dfs(int x)//求SG[x]模板
{
    if(SG[x]!=-1) return SG[x];
    bool vis[110];
    memset(vis,0,sizeof(vis));

    for(int i=0;i<k;i++)
    {
        if(x>=s[i])
        {
           dfs(x-s[i]);
           vis[SG[x-s[i]]]=1;
         }
    }
    int e;
    for(int i=0;;i++)
      if(!vis[i])
      {
        e=i;
        break;
      }
    return SG[x]=e;
}
int main()
{
    int cas,i;
    while(scanf("%d",&k)!=EOF)
    {
        if(!k) break;
        memset(SG,-1,sizeof(SG));
        for(i=0;i<k;i++) scanf("%d",&s[i]);
        sort(s,s+k);
        scanf("%d",&cas);
        while(cas--)
        {
            int t,sum=0;
            scanf("%d",&t);
            while(t--)
            {
                int num;
                scanf("%d",&num);
                sum^=dfs(num);
               // printf("SG[%d]=%d\n",num,SG[num]);
            }
            if(sum==0) printf("L");
            else printf("W");
        }
        printf("\n");
    }
    return 0;
}


//下面是对SG打表的做法
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
const int K=101;
const int H=10001;//H是我们要打表打到的最大值
int k,m,l,h,s[K],sg[H],mex[K];///k是集合元素的个数 s[]是集合  mex大小大约和集合大小差不多
///注意s的排序
void sprague_grundy()
{
    int i,j;
    sg[0]=0;
    for (i=1; i<H; i++)
    {
        memset(mex,0,sizeof(mex));
        j=1;
        while (j<=k && i>=s[j])
        {
            mex[sg[i-s[j]]]=1;
            j++;
        }
        j=0;
        while (mex[j]) j++;
        sg[i]=j;
    }
}

int main()
{
    int tmp,i,j;
    scanf("%d",&k);
    while (k!=0)
    {
        for (i=1; i<=k; i++)
            scanf("%d",&s[i]);
        sort(s+1,s+k+1);            //这个不能少
        sprague_grundy();
        scanf("%d",&m);
        for (i=0; i<m; i++)
        {
            scanf("%d",&l);
            tmp=0;
            for (j=0; j<l; j++)
            {
                scanf("%d",&h);
                tmp=tmp^sg[h];
            }
            if (tmp)
                putchar('W');
            else
                putchar('L');
        }
        putchar('\n');
        scanf("%d",&k);
    }
    return 0;
}

/*
    动态规划(dp):
*/






//O(nlogn)的LIS
int low[maxn];
int n,ans;
int binary_search(int *a,int r,int x)
{
    int l=1,mid;
    while(l<=r)
    {
        mid=(l+r)>>1;
        if(a[mid]<=x)
            l=mid+1;
        else
            r=mid-1;
    }
    return l;
}
int a[maxn];
int LIS(int tol)
{
    for(int i=1;i<=tol;i++)
    {
        low[i]=INF;
    }
    low[1]=a[1];
    ans=1;
    for(int i=2;i<=tol;i++)
    {
        if(a[i]>=low[ans])
            low[++ans]=a[i];
        else
            low[binary_search(low,ans,a[i])]=a[i];
    }
    cout<<ans<<endl;
}

//或者如下用lower_bound
#include<cstdio>
#include<algorithm>
const int MAXN=200001;
 
int a[MAXN];
int d[MAXN];
 
int main()
{
    int n;
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
        scanf("%d",&a[i]);
    d[1]=a[1];
    int len=1;
    for(int i=2;i<=n;i++)
    {
        if(a[i]>d[len])
            d[++len]=a[i];
        else
        {
            int j=std::lower_bound(d+1,d+len+1,a[i])-d;
            d[j]=a[i];
        }
    }
    printf("%d\n",len);   
    return 0;
}


//或者如下更简洁
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>
typedef long long ll;
using namespace std;
const int maxn=40009;
const int INF=0x3f3f3f3f;
int a[maxn];
int dp[maxn];
int main()
{
    int n;
    while(scanf("%d",&n) && n!=-1)
    {
        //输入
        for(int i=0;i<n;i++)
            scanf("%d",&a[i]);
        //nlogn 的最长升序子序列的解法
        memset(dp,INF,sizeof(dp));
        for(int i=0;i<n;i++)
        {
            *lower_bound(dp,dp+n,a[i])=a[i];
        }
        printf("%d\n",lower_bound(dp,dp+n,INF)-dp);
    }
    return 0;
}





//LCS O(nlogn) 适用范围：序列中每一个元素都不相同
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
#include<ext/rope>
using namespace std;
using namespace __gnu_cxx;
using namespace __gnu_pbds;
#define ll long long
#define ld long double
#define ull unsigned long long
#define rep(i,a,b) for(int i=a;i<b;i++)
#define Rep(i,a,b) for(int i=a;i<=b;i++)
#define per(i,a,b) for(int i=b-1;i>=a;i--)
#define Per(i,a,b) for(int i=b;i>=a;i--)
#define pb push_back
#define eb emplace_back
#define MP make_pair
#define fi first
#define se second
#define SZ(x) (x).size()
#define LEN(x) (x).length()
#define ALL(X) (X).begin(), (X).end()
#define MS0(X) memset((X), 0, sizeof((X)))
#define MS1(X) memset((X), -1, sizeof((X)))
#define MS(X,a) memset((X),a,sizeof(X))
#define CASET int ___T; scanf("%d", &___T); for(int cs=1;cs<=___T;cs++)
#define Read(x) scanf("%d",&x)
#define ReadD(x) scanf("%lf",&x)
#define ReadLL(x) scanf("%lld",&x)
#define ReadLD(x) scanf("%llf",&x)
#define Write(x) printf("%d\n",x)
#define WriteD(x) printf("%f\n",x)
#define WriteLL(x) printf("%lld\n",x)
#define WriteLD(x) printf("%Lf\n",x)
#define IO ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
constexpr ld pi=acos(-1);
constexpr ll mod=1e9+7;
#define lowbit(x) (x&(-x))
constexpr ld eps=1e-6;
constexpr int maxn=1e5+10;
constexpr int INF=0x3f3f3f3f;
constexpr double e=2.718281828459045;
typedef long long LL;
typedef unsigned long long ULL;
typedef long double LD;
typedef pair<int,int> PII;
typedef vector<int> VI;
typedef vector<LL> VL;
typedef vector<PII> VPII;
typedef pair<LL,LL> PLL;
typedef vector<PLL> VPLL;
typedef vector<int> VI;
typedef pair<int,int> PII;
#define Accepted 0
inline ll quick(ll a,ll b,ll m){ll ans=1;while(b){if(b&1)ans=(a*ans)%m;a=(a*a)%m;b>>=1;}return ans;}
inline ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
struct node
{
    int a,b;
}a[maxn];
bool cmp(node a,node b)
{
    return a.a<b.a;
}
int turn[maxn];
int d[maxn];
signed main()
{
    int n;
    Read(n);
    Rep(i,1,n)
    {
        Read(a[i].a);
    }
    Rep(i,1,n)
    {
        Read(a[i].b);
    }
    Rep(i,1,n)
    {
        turn[a[i].a]=i;
    }
    Rep(i,1,n)
    {
        a[i].b=turn[a[i].b];
    }
    d[1]=a[1].b;
    int len=1;
    for(int i=2;i<=n;i++)
    {
        if(a[i].b>d[len])
        {
            d[++len]=a[i].b;
        }
        else
        {
            int id=upper_bound(d+1,d+len+1,a[i].b)-d;
            d[id]=a[i].b;
        }
    }
    Write(len);
    return Accepted;
}








//区间dp板子 未用四边形优化   O(n^3)          P1880 石子合并
for(int len=1;len<=n-1;len++)             //枚举区间长度
{
    for(int i=1;i<n;i++)           //枚举区间左端点
    {
        int j=i+len;                      //区间右端点
        dp1[i][j]=INF;
        for(int k=i;k<j;k++)              //枚举断点
        {
            dp1[i][j]=min(dp1[i][j],dp1[i][k]+dp1[k+1][j]+a[j]-a[i-1]);           //合并[i,k],[k+1,j]所需的花费
            dp2[i][j]=max(dp2[i][j],dp2[i][k]+dp2[k+1][j]+a[j]-a[i-1]);
        }
    }
}












//树形dp   (HDU1561改编)
vector<int> E[maxn];
int val[maxn];
int cost[maxn];
int dp[maxn][maxn];
void dfs(int x,int w)
{
    for(int i=0;i<=w;i++)
    {
        if(i>=cost[x])
        {
            dp[x][i]=val[x];
        }
        else
        {
            dp[x][i]=0;
        }
    }
    //dp[i][j]的意义是以i为根节点的子树选取j个结点的最大价值
    //这个地方先枚举x的子节点i 对于前i个子树，枚举第i个子树的结点数量k
    //此时取决于对于前i-1个子节点取j-k个节点的状态
    for(int i:E[x])
    {
        dfs(i,w-cost[x]);
        for(int j=w;j>=cost[x];j--)
        {
            for(int k=0;k<=j-cost[x];k++)
            {
                dp[x][j]=max(dp[x][j],dp[x][j-k]+dp[i][k]);
            }
        }
    }
}






//数位dp模板    HDU2089
//求[n,m]中不含“4”和“62”的数的个数
#include<bits/stdc++.h>
using namespace std;
#define Accepted 0
int dist[10];
int dp[10][2];


/*
dp数组一定要初始化为-1
dp数组一定要初始化为-1
dp数组一定要初始化为-1
重要的事情说三遍
*/


//pre代表前一个数字是否为6，flag代表当前位是否有限制（是否以dist[len]结尾）
int dfs(int len,int pre,int flag)
{
    if(len<0)
        return 1;
    //如果当前询问的值已经被记忆化，直接返回
    if(!flag&&dp[len][pre]!=-1)
        return dp[len][pre];
    //判断当前位结尾
    int ed=flag?dist[len]:9;
    int ans=0;
    for(int i=0;i<=ed;i++)
    {
        if(i!=4&&!(pre&&i==2))
        {
            //如果当前位的end没有限制，那么递归下去的所有位都没有限制
            ans+=dfs(len-1,i==6,flag&&i==ed);
        }
    }
    //记忆化
    if(!flag)
    {
        dp[len][pre]=ans;
    }
    return ans;
}
//这个地方solve(0)也是有值的
//solve求的是[0,n]中满足题目要求的数的个数
int solve(int n)
{
    int len=0;
    while(n)
    {
        dist[len++]=n%10;
        n/=10;
    }
    return dfs(len-1,0,1);
}
signed main()
{
    ios::sync_with_stdio(false);
    int n,m;
    while(cin>>n>>m)
    {
        //这里一定要初始化为-1
        memset(dp,-1,sizeof(dp));
        if(!n&&!m)
            break;
        cout<<solve(m)-solve(n-1)<<endl;
    }
    return Accepted;
}


//数位dp可以做求[l,r]中 不包含某些子串的数的个数
//也可以不能求包含某些子串的数的个数
//如果要求，可约容斥原理+数位dp   如HDU3555

//或者参考如下HDU3652
//求[1,n]中包含13且%13==0的数的个数
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
#define Accepted 0
#define IO ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
const int maxn=1e5+10;
const double eps=1e-6;
const int INF=0x3f3f3f3f;
const double pi=acos(-1);
const int mod=1e9+7;
inline ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
inline ll quick(ll a,ll b,ll m){ll sum=1;while(b){if(b&1)sum=(sum*a)%m;a=(a*a)%m;b>>=1;}return sum;}
int dp[50][10][20];
int dist[50];
int dfs(int len,int pre,int md,int flag)
{
    if(len<0)
    {
        return pre==2&&md==0;
    }
    if(!flag&&dp[len][pre][md]!=-1)
    {
        return dp[len][pre][md];
    }
    int ed=flag?dist[len]:9;
    int ans=0;
    for(int i=0;i<=ed;i++)
    {
        if(pre==2||(pre==1&&i==3))
        {
            ans+=dfs(len-1,2,(md*10+i)%13,flag&&i==ed);
        }
        else if(i==1)
        {
            ans+=dfs(len-1,1,(md*10+i)%13,flag&&i==ed);
        }
        else
        {
            ans+=dfs(len-1,0,(md*10+i)%13,flag&&i==ed);
        }
        
    }
    if(!flag)
    {
        dp[len][pre][md]=ans;
    }
    return ans;
}
int solve(int n)
{
    int len=0;
    while(n)
    {
        dist[len++]=n%10;
        n/=10;
    }
    return dfs(len-1,0,0,1);
}
signed main()
{
    IO
    int n;
    while(cin>>n)
    {
        memset(dp,-1,sizeof(dp));
        cout<<solve(n)<<endl;
    }
    return 0;
}









//单调队列优化dp   （洛谷P3957跳房子）
int a[maxn];
int s[maxn];
ll dp[maxn];
ll k;
int n,d,ans=-1;
int q[maxn];        //q用来记录下标，在q[l]~q[r]内的下标都是合法的
bool check(int g)
{
    //cout<<"g="<<g<<endl;
    for(int i=1; i<=n; i++)
    {
        dp[i]=-1e18;
    }
    int x1=g+d,x2=max(d-g,1);
    int now=0;
    int l=0,r=-1;     //[l,r]记录队列的区间 初始状态l>r队列为空
    int i;
    for(i=1; i<=n; i++)
    {
        int L=a[i]-x1,R=a[i]-x2; //L,R用来记录合法范围
        //cout<<L<<' '<<R<<endl;
        while(a[now]<=R&&now<i)    //Insert(now)这个点
        {
            while(r>=l&&dp[q[r]]<=dp[now])r--;   //维护最大值，就保证是一个单调递减的单调队列
            q[++r]=now;
            now++;
        }
        while(a[q[l]]<L&&l<=r)l++;            //pop_front 把队列前端不合法的数去掉
        if(l>r||dp[q[l]]==-1e18)
            continue;
        dp[i]=dp[q[l]]+s[i];
        //cout<<a[i]<<' '<<dp[i]<<endl;
        if(dp[i]>=k)
            return true;
    }
    return false;
}




//单调队列优化dp (HDU板子)
typedef pair<long long,long long> P;
const int maxn=5010,maxk=2010;
struct Clique
{
    P q[maxn];
    int top,tail;
 
    void Init() {top=1,tail=0;}
 
    void push(long long k,long long b)
    {
        while (tail>top && (__int128)(b-q[tail-1].second)*(q[tail].first-q[tail-1].first)>=(__int128)(q[tail].second-q[tail-1].second)*(k-q[tail-1].first)) tail--;
        ++tail;
        q[tail].first=k;
        q[tail].second=b;
    }
 
    void autopop(long long x)
    {
        while (top<tail && q[top].first*x+q[top].second<=q[top+1].first*x+q[top+1].second) top++;
    }
 
    P front() {return q[top];}
 
}Q[maxk];

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41136357



struct Clique
{
    #define P pair<long long,long long>
    P q[maxn];
    //top是队首结点，tail为队尾结点
    int top,tail;
    Clique(){top=1,tail=0;}
    void Init() {top=1,tail=0;}
    //加入一个结点时,删除部分不优的结点
    void push(long long f,long long s)
    {
        while (tail>=top && s>q[tail].second)
            tail--;
        ++tail;
        q[tail].first=f;
        q[tail].second=s;
    }
    //在查询之前，弹出已经越界的不符合要求的结点
    void autopop(long long x)
    {
        while (top<=tail && q[top].first<=x) top++;
    }
    ll query(ll x)
    {
        int l=top,r=tail,ans=-1,mid;
        while(l<=r)
        {
            mid=(l+r)>>1;
            if(q[mid].second>=x)
            {
                ans=q[mid].first;
                l=mid+1;
            }
            else
                r=mid-1;
        }
        return ans;
    }
    P front() {return q[top];}
    #undef P
}Q;














//斜率优化dp板子
struct Line{
    ll k,b;
    ll f(ll x){
        return k*x+b;
    }
};
struct Hull{
    vector<Line>ve;
    int cnt,idx;
    bool empty(){return cnt==0;}
    void init(){ve.clear();cnt=idx=0;}
    void add(const Line& p){ve.push_back(p);cnt++;}
    void pop(){ve.pop_back();cnt--;}
    bool checkld(const Line& a,const Line& b,const Line& c){
        return (long double)(a.b-b.b)/(long double)(b.k-a.k)>(long double)(a.b-c.b)/(long double)(c.k-a.k);
    }
    bool checkll(const Line& a,const Line& b,const Line& c){
        return (a.b-b.b)*(c.k-a.k)>(a.b-c.b)*(b.k-a.k);
    }
    void insert(const Line& p){
        if(cnt&&ve.back().k==p.k){
            if(p.b<=ve.back().b)return;
            else pop();
        }
        while(cnt>=2&&checkld(ve[cnt-2],ve[cnt-1],p))pop();
        add(p);
    }
    ll query(ll x){
        while(idx+1<cnt&&ve[idx+1].f(x)>ve[idx].f(x))idx++;
        return ve[idx].f(x);
    }
}hull[2005];

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41129697





//斜率优化dp  quailty
struct Line {
    mutable ll k,m,p;
    bool operator <(const Line& o)const { return k<o.k; }
    bool operator <(ll x)const { return p<x; }
};
struct LineContainer : multiset<Line,less<> > {
    const ll inf = LLONG_MAX;
    ll div(ll a,ll b){
        return a/b-((a^b)<0&&a%b);
    }
    bool isect(iterator x,iterator y){
        if (y==end()){x->p=inf; return false; }
        if (x->k==y->k)x->p=x->m>y->m?inf:-inf;
        else x->p=div(y->m-x->m,x->k-y->k);
        return x->p>=y->p;
    }
    void add(ll k,ll m) {
        auto z=insert({k,m,0}),y=z++,x=y;
        while (isect(y,z))z=erase(z);
        if (x!=begin() && isect(--x,y))isect(x,y=erase(y));
        while ((y=x)!=begin() && (--x)->p>=y->p)
            isect(x,erase(y));
    }
    ll query(ll x){
        assert(!empty());
        auto l=*lower_bound(x);
        return l.k*x+l.m;
    }
}h;

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41130719

// Created by calabash_boy on 18/6/5.
// HDU 6138
//给定若干字典串。
// query:strx stry 求最长的p,p为strx、stry子串，且p为某字典串的前缀
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+100;
struct Aho_Corasick_Automaton
{
//basic
    int nxt[maxn*10][26],fail[maxn*10];
    int root,tot;
//special
    int flag[maxn*10];
    int len[maxn*10];
    void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        flag[tot] = len[tot]=0;
        return tot;
    }
    void insert(char *s )
    {
        int now = root;
        while (*s)
        {
            int id = *s-'a';
            if(!nxt[now][id])nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now]+1;
            now = nxt[now][id];
        }
    }
    void insert(string str)
    {
        int now = root;
        for (int i=0; i<str.size(); i++)
        {
            int id = str[i]-'a';
            if(!nxt[now][id])nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now]+1;
            now = nxt[now][id];
        }
    }
    void build()
    {
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (int i=0; i<26; i++)
            {
                if(!nxt[head][i])continue;
                int temp = nxt[head][i];
                fail[temp] = fail[head];
                while (fail[temp]&&!nxt[fail[temp]][i])
                {
                    fail[temp] = fail[fail[temp]];
                }
                if(head&&nxt[fail[temp]][i])fail[temp] = nxt[fail[temp]][i];
                Q.push(temp);
            }
        }
    }
    void search(string str,int QID);
    int query(string str,int QID);
} acam;
void Aho_Corasick_Automaton::search(string str,int QID)
{
    int now = root;
    for (int i=0; i<str.size(); i++)
    {
        int id = str[i]-'a';
        now = nxt[now][id];
        int temp = now;
        while (temp!=root&&flag[temp]!=QID)
        {
            flag[temp] = QID;
            temp = fail[temp];
        }
    }
}
int Aho_Corasick_Automaton::query(string str, int QID)
{
    int ans =0;
    int now = root;
    for (int i=0; i<str.size(); i++)
    {
        int id = str[i]-'a';
        now = nxt[now][id];
        int temp = now;
        while (temp!=root)
        {
            if(flag[temp]==QID)
            {
                ans = max(ans,len[temp]);
                break;
            }
            temp = fail[temp];
        }
    }
    return ans;
}
string a[maxn];
int m,n,qid;
int main()
{
    int T;
    cin>>T;
    while (T--)
    {
        acam.clear();
        cin>>n;
        for (int i=1; i<=n; i++)
        {
            cin>>a[i];
            acam.insert(a[i]);
        }
        acam.build();
        cin>>m;
        for (int i=1; i<=m; i++)
        {
            int x,y;
            cin>>x>>y;
            qid++;
            acam.search(a[x],qid);
            int ans = acam.query(a[y],qid);
            cout<<ans<<endl;
        }
    }
    return 0;
}






//AC自动机 by:LZW  HDU2896
#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<vector>
#include<string>
#include<list>
#include<stack>
#include<queue>
#include<deque>
#include<map>
#include<set>
#include<bitset>
#include<utility>
#include<iomanip>
#include<climits>
#include<complex>
#include<cassert>
#include<functional>
#include<numeric>
#define Accepted 0
typedef long long ll;
typedef long double ld;
const int mod=1e9+7;
const int maxn=1e5+10;
const double pi=acos(-1);
const double eps=1e-6;
const int INF=0x3f3f3f3f;
using namespace std;
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}
ll lowbit(ll x)
{
    return x&(-x);
}

struct Aho_Corasick_Automaton
{
#define MAX maxn
#define type int
/***********************/
//basic
    //nxt为构建字典树的数组,fail维护失配时的转移后的结点的下标
    int nxt[MAX][100],fail[MAX];
    int root,tot;        //根节点下标和字典树中结点的数量
//special 结点中维护的信息，和结点同步更新
    type v[MAX];
/***********************/
    int getid(char c)
    {
        return c-32;     //may change
    }
    void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    //从内存池中新建一个结点
    int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        //////
        v[tot]=0;
        return tot;
    }
    //向Trie中插入一个字符串
    void insert(string str,int num)
    {
        int now = root;
        int len=str.size();
        for (int i=0; i<len; i++)
        {
            int id = getid(str[i]);
            if(!nxt[now][id])nxt[now][id] = newnode();
            now = nxt[now][id];
            if(i==len-1)
                v[now]=num;
        }
    }
    //BFS建立fail指针
    void build()
    {
        //root的fail为自己
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (int i=0; i<100; i++)
            {
                if(!nxt[head][i])continue;

                int temp = nxt[head][i];
                fail[temp] = fail[head];
                //若temp的fail没有到达root,且temp当前的fail位置的下一个位置的对应结点为空
                //则temp的fail再向前移动，这里的转移可以结合kmp算法理解
                while (fail[temp]&&!nxt[fail[temp]][i])
                {
                    fail[temp] = fail[fail[temp]];
                }
                //如果temp的fail位置的下一个对应结点存在，则直接赋值
                if(head&&nxt[fail[temp]][i])fail[temp] = nxt[fail[temp]][i];
                Q.push(temp);
            }
        }
    }
    int query(string str,int num);
#undef type
#undef MAX
} ac;

int Aho_Corasick_Automaton::query(string str,int num)
{
    set<int> s;
    int ans = 0;
    int now = root,len=str.length();
    for (int i=0; i<len; i++)
    {
        int id = getid(str[i]);
        while(now&&!nxt[now][id])
        {
            now=fail[now];
        }
        int temp=nxt[now][id];
        while(temp)
        {
            if(v[temp])
                s.insert(v[temp]);
            temp=fail[temp];
        }
        now=nxt[now][id];
    }
    if(s.size()>0)
    {
        cout<<"web "<<num<<':';
        for(int c:s)
        {
            cout<<' '<<c;
        }
        cout<<endl;
        return 1;
    }
    else
        return 0;
}
string str;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int n;
    cin>>n;
    for (int i=1; i<=n; i++)
    {
        cin>>str;
        ac.insert(str,i);
    }
    /*****别忘记build*****/
    ac.build();
    int m;
    cin>>m;
    int ans=0;
    for(int i=1;i<=m;i++)
    {
        cin>>str;
        ans+=ac.query(str,i);
    }
    cout<<"total: "<<ans<<endl;
    return 0;
}

//Treap部分
//PS:按size来split可以达到区间操作，按value可达到集合操作

//不重复随机数生成函数               (Treap)
inline int Rand(){
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}

//还是随机数生成
int Seed = 19260817 ;
inline int Rand() {
    Seed ^= Seed << 7 ;
    Seed ^= Seed >> 5 ;
    Seed ^= Seed << 13 ;
    return Seed ;
}


//无旋Treap              （普通平衡树）
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-9;
const int maxn=1e5+10;
#define Accepted 0
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
#define ls(x) arr[x].child[0]
#define rs(x) arr[x].child[1]
inline int Rand(){                                 //随机数
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node                                     //Treap结点
{
    int child[2],value,key,size;
}arr[maxn];
int tot;                                       //结点数量
inline void Push_Up(int x)                       //更新size
{
    arr[x].size=arr[ls(x)].size+arr[rs(x)].size+1;
}
void Split(int root,int& x,int& y,int value)         //按值切分无旋Treap树
{
    if(!root)
    {
        x=y=0;
        return ;
    }
    if(arr[root].value<=value) x=root,Split(rs(root),rs(x),y,value);
    else y=root,Split(ls(root),x,ls(y),value);
    Push_Up(root);
}
void Merge(int& root,int x,int y)                   //合并两个无旋Treap子树
{
    if(!x||!y)
    {
        root=x+y;
        return ;
    }
    if(arr[x].key<arr[y].key)root=x,Merge(rs(root),rs(x),y);
    else root=y,Merge(ls(root),x,ls(y));
    Push_Up(root);
}
void Insert(int& root,int value)                  //插入结点
{
    int x=0,y=0,z=++tot;
    arr[z].key=Rand(),arr[z].size=1,arr[z].value=value;
    Split(root,x,y,value);
    Merge(x,x,z);
    Merge(root,x,y);
}
void Erase(int& root,int value)                //删除结点
{
    int x=0,y=0,z=0;
    Split(root,x,y,value);
    Split(x,x,z,value-1);
    Merge(z,ls(z),rs(z));
    Merge(x,x,z);
    Merge(root,x,y);
}
int Kth_number(int root,int k)               //树上第k小
{
    while(arr[ls(root)].size+1!=k)
    {
        if(arr[ls(root)].size>=k)root=ls(root);
        else k-=arr[ls(root)].size+1,root=rs(root);
    }
    return arr[root].value;
}
int Get_rank(int& root,int value)             //树上名次
{
    int x=0,y=0;
    Split(root,x,y,value-1);
    int res=arr[x].size+1;
    Merge(root,x,y);
    return res;
}
int Pre(int& root,int value)                //value的前驱
{
    int x=0,y=0;
    Split(root,x,y,value-1);
    int res=Kth_number(x,arr[x].size);
    Merge(root,x,y);
    return res;
}
int Suf(int& root,int value)               //value的后继
{
    int x=0,y=0;
    Split(root,x,y,value);
    int res=Kth_number(y,1);
    Merge(root,x,y);
    return res;
}
int root;                                //root=0,初始化空树
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int n;
    cin>>n;
    while(n--)
    {
        int opt,x;
        cin>>opt>>x;
        if(opt==1)
            Insert(root,x);
        else if(opt==2)
            Erase(root,x);
        else if(opt==3)
            cout<<Get_rank(root,x)<<endl;
        else if(opt==4)
            cout<<Kth_number(root,x)<<endl;
        else if(opt==5)
            cout<<Pre(root,x)<<endl;
        else if(opt==6)
            cout<<Suf(root,x)<<endl;
    }
    return Accepted;
}







//文艺平衡树 无旋Treap实现
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-9;
const int maxn=1e5+10;
#define Accepted 0
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}
#define ls(x) arr[x].child[0]
#define rs(x) arr[x].child[1]
inline int Rand()                                  //随机数
{
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node                                     //Treap结点
{
    int child[2],value,key,size,tag;
} arr[maxn];
int tot;                                       //结点数量
inline void Push_Up(int x)
{
    arr[x].size=arr[ls(x)].size+arr[rs(x)].size+1;
}

inline void Push_Down(int x)                  //下推翻转标记
{
    if(arr[x].tag)
    {
        arr[ls(x)].tag^=1;
        arr[rs(x)].tag^=1;
        swap(ls(x),rs(x));
        arr[x].tag^=1;
    }
}

void Split(int root,int &x,int &y,int Sz)           //按Size切分
{
    if(!root)
    {
        x=y=0;
        return ;
    }
    Push_Down(root);
    if(Sz<=arr[ls(root)].size)
        y=root,Split(ls(root),x,ls(y),Sz);
    else x=root,Split(rs(root),rs(root),y,Sz-arr[ls(root)].size-1);
    Push_Up(root);
}

void Merge(int&root,int x,int y)             //合并子树
{
    if(!x||!y)
    {
        root=x+y;
        return;
    }
    Push_Down(x),Push_Down(y);
    if(arr[x].key<arr[y].key)root=x,Merge(rs(root),rs(x),y);
    else root=y,Merge(ls(root),x,ls(y));
    Push_Up(root);
}

inline void Insert(int&root,int value)          //插入一个结点
{
    int x=0,y=0,z=++tot;
    arr[z].size=1,arr[z].value=value,arr[z].key=Rand(),arr[z].tag=0;
    Split(root,x,y,value);
    Merge(x,x,z);
    Merge(root,x,y);
}

void Rever(int&root,int L,int R)                 //翻转一个区间
{
    int x=0,y=0,z=0,t=0;
    Split(root,x,y,R);
    Split(x,z,t,L-1);
    arr[t].tag^=1;
    Merge(x,z,t);
    Merge(root,x,y);
}
void Print(int x)                             //中序遍历输出
{
    if(!x)return ;
    Push_Down(x);
    Print(ls(x));
    cout<<arr[x].value<<' ';
    Print(rs(x));
}
int root;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int n,m;
    cin>>n>>m;
    for(int i=1; i<=n; i++)
        Insert(root,i);
    while(m--)
    {
        int a,b;
        cin>>a>>b;
        Rever(root,a,b);
    }
    Print(root);
    return Accepted;
}



//指针版Treap 省内存
// It is made by XZZ
#include<cstdio>
#include<algorithm>
#define pr pair<point,point>
#define mp make_pair
using namespace std;
#define rep(a,b,c) for(rg int a=b;a<=c;a++)
#define drep(a,b,c) for(rg int a=b;a>=c;a--)
#define erep(a,b) for(rg int a=fir[b];a;a=nxt[a])
#define il inline
#define rg register
#define vd void
typedef long long ll;
il int gi(){
    rg int x=0,f=1;rg char ch=getchar();
    while(ch<'0'||ch>'9')f=ch=='-'?-1:f,ch=getchar();
    while(ch>='0'&&ch<='9')x=x*10+ch-'0',ch=getchar();
    return x*f;
}
int seed=19260817;
il int Rand(){return seed=seed*48271ll%2147483647;}
typedef struct node* point;
point null;
struct node{
    char data;
    int size,rand;
    point ls,rs;
    bool rev;
    node(char ch){data=ch,size=1,rand=Rand(),rev=0,ls=rs=null;}
    il vd down(){if(rev)rev=0,ls->rev^=1,rs->rev^=1,swap(ls,rs);}
    il vd reset(){if(ls!=null)ls->down();if(rs!=null)rs->down();size=ls->size+rs->size+1;}
};
point root=null;
il point build(int n){
    point stack[n+1];
    int top=0;
    char ch;
    rep(i,1,n){
    ch=getchar();while(ch=='\n')ch=getchar();
    point now=new node(ch),lst=null;
    while(top&&stack[top]->rand>now->rand)lst=stack[top],stack[top--]->reset();
    now->ls=lst;if(top)stack[top]->rs=now;stack[++top]=now;
    }
    while(top)stack[top--]->reset();
    return stack[1];
}
il point merge(point a,point b){
    if(a==null)return b;
    if(b==null)return a;
    if(a->rand<b->rand){a->down(),a->rs=merge(a->rs,b),a->reset();return a;}
    else {b->down(),b->ls=merge(a,b->ls),b->reset();return b;}
}
il pr split(point now,int num){
    if(now==null)return mp(null,null);
    now->down();
    point ls=now->ls,rs=now->rs;
    if(num==ls->size){now->ls=null,now->reset();return mp(ls,now);}
    if(num==ls->size+1){now->rs=null,now->reset();return mp(now,rs);}
    if(num<ls->size){pr T=split(ls,num);now->ls=T.second,now->reset();return mp(T.first,now);}
    pr T=split(rs,num-ls->size-1);now->rs=T.first,now->reset();return mp(now,T.second);
}
il vd del(point now){if(now!=null)del(now->ls),del(now->rs),delete now;}
int main(){
    int m=gi()-1,pos=0;
    char opt[10];
    null=new node('%');
    null->size=0;
    {
    scanf("%*s");
    root=build(gi());
    }
    while(m--){
    scanf("%s",opt);
    if(opt[0]=='M')pos=max(0,min(gi(),root->size));
    else if(opt[0]=='P'){if(pos)--pos;}
    else if(opt[0]=='N'){if(pos!=root->size)++pos;}
    else if(opt[0]=='G'){
        pr T=split(root,pos);
        char lst;point now=T.second;
        while(now!=null)lst=now->data,now->down(),now=now->ls;
        printf("%c\n",lst);
        root=merge(T.first,T.second);
    }
    else if(opt[0]=='I'){
        pr T=split(root,pos);
        root=merge(T.first,merge(build(gi()),T.second));
    }
    else{
        pr T=split(root,pos),TT=split(T.second,gi());
        if(opt[0]=='D')root=merge(T.first,TT.second),del(TT.first);
        else TT.first->rev^=1,root=merge(T.first,merge(TT.first,TT.second));
    }
    }
    del(root);
    delete null;
    return 0;
}







//Treap
struct node{ //节点数据的结构
    int key,prio,size; //size是指以这个节点为根的子树中节点的数量
    node* ch[2]; //ch[0]指左儿子，ch[1]指右儿子
};

typedef node* tree;

node base[MAXN],nil;
tree top,null,root;

void init(){                //初始化top和null
    top=base;
    root=null=&nil;
    null->ch[0]=null->ch[1]=null;
    null->key=null->prio=2147483647;
    null->size=0;
}

inline tree newnode(int k){ //注意这种分配内存的方法也就比赛的时候用用，仅仅是为了提高效率
    top->key=k;				//看为val,是BST的键值
    top->size=1;
    top->prio=random();
    top->ch[0]=top->ch[1]=null;
    return top++;
}


//Treap结点的旋转
void rotate(tree &x,bool d){ //d指旋转的方向，0为左旋，1为右旋
    tree y=x->ch[!d];        //x为要旋的子树的根节点
    x->ch[!d]=y->ch[d];
    y->ch[d]=x;
    x->size=x->ch[0]->size+1+x->ch[1]->size;
    y->size=y->ch[0]->size+1+y->ch[1]->size;
    x=y;
}

void insert(tree &t,int key){ //插入一个节点
    if (t==null) t=newnode(key);
    else{
        bool d=key>t->key;
        insert(t->ch[d],key);
        t->size++;
        if (t->prio<t->ch[d]->prio) rotate(t,!d);
    }
}

void erase(tree &t,int key){ //删除一个节点
    if (t->key!=key){
        erase(t->ch[key>t->key],key);
        t->size--;
	}
    else if (t->ch[0]==null) t=t->ch[1];
    else if (t->ch[1]==null) t=t->ch[0];
    else{
        bool d=t->ch[0]->prio<t->ch[1]->prio;
        rotate(t,d);
        erase(t->ch[d],key);
    }
}

tree select(int k){ //选择第k小节点
    tree t=root;
    for (int tmp;;){
        tmp=t->ch[0]->size+1;
        if (k==tmp) return t;
        if (k>tmp){
            k-=tmp;
            t=t->ch[1];
        }
        else t=t->ch[0];
    }
}









//Treap   by:sjf
struct Treap
{
	#define type ll
	struct node
	{
		int ch[2],fix,sz,w;
		type v;
		node(){}
		node(type x)
		{
			v=x;
			fix=rand();
			sz=w=1;
			ch[0]=ch[1]=0;
		} 
	}t[MAX];  
	int tot,root,tmp;
	void init()
	{
		srand(unsigned(new char));
		root=tot=0;
		t[0].sz=t[0].w=0;
		mem(t[0].ch,0);
	}
	inline void maintain(int k)  
	{  
		t[k].sz=t[t[k].ch[0]].sz+t[t[k].ch[1]].sz+t[k].w ;  
	}  
	inline void rotate(int &id,int k)
	{
		int y=t[id].ch[k^1];
		t[id].ch[k^1]=t[y].ch[k];
		t[y].ch[k]=id;
		maintain(id);
		maintain(y);
		id=y;
	}
	void insert(int &id,type v)
	{
		if(!id) t[id=++tot]=node(v);
		else
		{
			if(t[id].sz++,t[id].v==v)t[id].w++;
			else if(insert(t[id].ch[tmp=v>t[id].v],v),t[t[id].ch[tmp]].fix>t[id].fix) rotate(id,tmp^1);
	    }
	}
	void erase(int &id,type v)
	{
		if(!id)return;
		if(t[id].v==v)
		{
			if(t[id].w>1) t[id].w--,t[id].sz--;
			else
			{
				if(!(t[id].ch[0]&&t[id].ch[1])) id=t[id].ch[0]|t[id].ch[1];
				else
				{
					rotate(id,tmp=t[t[id].ch[0]].fix>t[t[id].ch[1]].fix);
					t[id].sz--;
					erase(t[id].ch[tmp],v);
				}
			}
		}
		else
		{
			t[id].sz--;
			erase(t[id].ch[v>t[id].v],v);
		}
	}
	type kth(int k)//k small
	{
		int id=root;
		if(id==0) return 0;
		while(id)
		{
			if(t[t[id].ch[0]].sz>=k) id=t[id].ch[0];
			else if(t[t[id].ch[0]].sz+t[id].w>=k) return t[id].v;
			else
			{
				k-=t[t[id].ch[0]].sz+t[id].w;
				id=t[id].ch[1];
			}
		}
	}
	int find(type key,int f)
	{
		int id=root,res=0;
		while(id)
		{
			if(t[id].v<=key)
			{
				res+=t[t[id].ch[0]].sz+t[id].w;
				if(f&&key==t[id].v) res-=t[id].w;
				id=t[id].ch[1];
			}
			else id=t[id].ch[0];
		}
		return res;
	}
	type find_pre(type key)
	{
		type res=-LLINF;
		int id=root;
		while(id)
		{
			if(t[id].v<key)
			{
				res=max(res,t[id].v);
				id=t[id].ch[1];
			}
			else id=t[id].ch[0];
		}
		return res;
	}
	type find_suc(type key)
	{
		type res=LLINF;
		int id=root;
		while(id)
		{
			if(t[id].v>key)
			{
				res=min(res,t[id].v);
				id=t[id].ch[0];
			}
			else id=t[id].ch[1];
		}
		return res;
	}
	void insert(type v){insert(root,v);}
	void erase(type v){erase(root,v);}
	int upper_bound_count(type key){return find(key,0);}//the count >=key
	int lower_bound_count(type key){return find(key,1);}//the count >key
	int rank(type key){return lower_bound_count(key)+1;}
	#undef type
}t; //t.init();

//线段树板子2 SegmentTree
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
bool Finish_read;
template<class T>inline void read(T &x)
{
    Finish_read=0;
    x=0;
    int f=1;
    char ch=getchar();
    while(!isdigit(ch))
    {
        if(ch=='-')f=-1;
        if(ch==EOF)return;
        ch=getchar();
    }
    while(isdigit(ch))x=x*10+ch-'0',ch=getchar();
    x*=f;
    Finish_read=1;
}
template<class T>inline void print(T x)
{
    if(x/10!=0)print(x/10);
    putchar(x%10+'0');
}
template<class T>inline void writeln(T x)
{
    if(x<0)putchar('-');
    x=abs(x);
    print(x);
    putchar('\n');
}
template<class T>inline void write(T x)
{
    if(x<0)putchar('-');
    x=abs(x);
    print(x);
}
/*================Header Template==============*/
const int maxn=2e5+5;
#define ls(o) o<<1
#define rs(o) (o<<1|1)
int n,m;
ll p;
int a[maxn];
struct node
{
    int l,r,sz;
    ll val;
    ll addtag,multag;
} t[maxn<<2];
/*==================Define Area================*/
namespace SegmentTree
{
void update(int o)
{
    t[o].val=(t[ls(o)].val+t[rs(o)].val)%p;
}

void pushdown(int o)
{
    if(t[o].multag!=1)
    {
        t[ls(o)].multag*=t[o].multag;
        t[ls(o)].multag%=p;
        t[rs(o)].multag*=t[o].multag;
        t[rs(o)].multag%=p;
        t[ls(o)].addtag*=t[o].multag;
        t[ls(o)].addtag%=p;
        t[rs(o)].addtag*=t[o].multag;
        t[rs(o)].addtag%=p;
        t[ls(o)].val*=t[o].multag;
        t[ls(o)].val%=p;
        t[rs(o)].val*=t[o].multag;
        t[rs(o)].val%=p;
        t[o].multag=1;
    }
    if(t[o].addtag)
    {
        t[ls(o)].addtag+=t[o].addtag;
        t[ls(o)].addtag%=p;
        t[rs(o)].addtag+=t[o].addtag;
        t[rs(o)].addtag%=p;
        t[ls(o)].val+=t[o].addtag*t[ls(o)].sz;
        t[ls(o)].val%=p;
        t[rs(o)].val+=t[o].addtag*t[rs(o)].sz;
        t[rs(o)].val%=p;
        t[o].addtag=0;
    }
    return ;
}

void Build(int o,int l,int r)
{
    t[o].l=l,t[o].r=r;
    t[o].sz=r-l+1;
    t[o].multag=1;
    if(t[o].l==t[o].r)
    {
        t[o].val=a[l];
        return ;
    }
    int mid=(l+r)>>1;
    Build(ls(o),l,mid);
    Build(rs(o),mid+1,r);
    update(o);
}

ll IntervalSum(int o,int l,int r)
{
    if(t[o].l>r||t[o].r<l) return 0;
    if(t[o].l>=l&&t[o].r<=r)
    {
        pushdown(o);
        return t[o].val;
    }
    pushdown(o);
    ll ans=0;
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) ans+=IntervalSum(ls(o),l,r),ans%=p;
    if(mid<=r) ans+=IntervalSum(rs(o),l,r),ans%=p;
    return ans;
}

void IntervalAdd(int o,int l,int r,int v)
{
    if(t[o].l>r||t[o].r<l) return ;
    if(t[o].l>=l&&t[o].r<=r)
    {
        t[o].val+=t[o].sz*v;
        t[o].addtag+=v;
        t[o].addtag%=p;
        t[o].val%=p;
        return ;
    }
    pushdown(o);
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) IntervalAdd(ls(o),l,r,v);
    if(mid<=r) IntervalAdd(rs(o),l,r,v);
    update(o);
    return ;
}

void IntervalMul(int o,int l,int r,int v)
{
    if(t[o].l>r||t[o].r<l) return ;
    if(t[o].l>=l&&t[o].r<=r)
    {
        pushdown(o);
        t[o].val*=v;
        t[o].val%=p;
        t[o].addtag*=v;
        t[o].addtag%=p;
        t[o].multag*=v;
        t[o].multag%=p;
        return ;
    }
    pushdown(o);
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) IntervalMul(ls(o),l,r,v);
    if(mid<=r) IntervalMul(rs(o),l,r,v);
    update(o);
    return ;
}
}
using namespace SegmentTree;

int main()
{
    read(n);
    read(m);
    read(p);
    for(int i=1; i<=n; i++)
    {
        read(a[i]);
    }
    Build(1,1,n);
    for(int i=1; i<=m; i++)
    {
        int opt;
        read(opt);
        if(opt==1)
        {
            int x,y,k;
            read(x);
            read(y);
            read(k);
            IntervalMul(1,x,y,k);
        }
        if(opt==2)
        {
            int x,y,k;
            read(x);
            read(y);
            read(k);
            IntervalAdd(1,x,y,k);
        }
        if(opt==3)
        {
            int x,y;
            read(x);
            read(y);
            ll ans=IntervalSum(1,x,y);
            printf("%lld\n",ans);
        }
    }
    return 0;
}

#include <iostream>
#include <cstdio>
#include <cstring>
#include <stack>
#include <queue>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
const double eps = 1e-6;
const double pi = acos(-1.0);
const int INF = 0x3f3f3f3f;
const int MOD = 1000000007;
#define ll long long
#define CL(a,b) memset(a,b,sizeof(a))
#define MAXN 100010
 
struct node
{
    int l,r;
    ll s,add;//add为每次加的数
}t[MAXN<<2];
int hh[MAXN];
int n,q;
ll ans;
 
void build(int l, int r, int i)
{
    t[i].l = l;
    t[i].r = r;
    t[i].add = 0;
    if(l == r) return ;
    int mid = (l+r)>>1;
    build(l, mid, i<<1);
    build(mid+1, r, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
void update(int l, int r, int add, int i)
{
    if(t[i].l>r || t[i].r<l) return ;
    if(t[i].l>=l && t[i].r<=r)
    {
        t[i].s += (t[i].r-t[i].l+1)*add;
        t[i].add += add;
        return ;
    }
    if(t[i].add)
    {
        t[i<<1].s += (t[i<<1].r-t[i<<1].l+1)*t[i].add;
        t[i<<1].add += t[i].add;
        t[i<<1|1].s += (t[i<<1|1].r-t[i<<1|1].l+1)*t[i].add;
        t[i<<1|1].add += t[i].add;
        t[i].add = 0;
    }
    update(l, r, add, i<<1);
    update(l, r, add, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
void query(int l, int r, int i)
{
    if(t[i].l>r || t[i].r<l) return ;
    if(t[i].l>=l && t[i].r<=r)
    {
        ans += t[i].s;
        return ;
    }
    if(t[i].add)
    {
        t[i<<1].s += (t[i<<1].r-t[i<<1].l+1)*t[i].add;
        t[i<<1].add += t[i].add;
        t[i<<1|1].s += (t[i<<1|1].r-t[i<<1|1].l+1)*t[i].add;
        t[i<<1|1].add += t[i].add;
        t[i].add = 0;
    }
    query(l, r, i<<1);
    query(l, r, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
int main()
{
    int a,b,c;
    ll k;
    char ch;
    while(scanf("%d%d",&n,&q)==2)
    {
        for(int i=1; i<=n; i++)
            scanf("%d",&hh[i]);
        build(1, n, 1);
        for(int i=1; i<=n; i++)
            update(i, i, hh[i], 1);
        while(q--)
        {
            getchar();
            scanf("%c",&ch);
            if(ch == 'C')
            {
                scanf("%d%d%d",&a,&b,&c);
                update(a, b, c, 1);
            }
            if(ch == 'Q')
            {
                ans = 0;
                scanf("%d%d",&a,&b);
                query(a, b, 1);
                printf("%lld\n",ans);
            }
        }
    }
    return 0;
}



//还是线段树板子 维护区间和，区间平方和，支持区间修改
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
struct node
{
    ll sum;//当前节点所表示的区间的和
    ll asign;//加法延迟标记
    ll msign;//乘法延迟标记
    ll sq;
};
ll a[10009];//以此数组建树
ll n,m;//数组的大小,取模,询问次数
node t[4*10009];
void build(int root,int l,int r)//build(1,1,n)进行建树
{
    t[root].msign=1;
    if(l==r)
    {t[root].sum=a[l];
    t[root].sq=a[l]*a[l];
    return ;}
    int mid=(l+r)>>1;
    build(root<<1,l,mid);
    build(root<<1|1,mid+1,r);
    t[root].sum=t[root<<1].sum+t[root<<1|1].sum;
    t[root].sq=t[root<<1].sq+t[root<<1|1].sq;
}
void push_down(int rt,int l,int r)
{
    if(t[rt].msign!=1)
    {
        t[rt<<1].msign*=t[rt].msign;
        t[rt<<1].asign*=t[rt].msign;
        t[rt<<1|1].msign*=t[rt].msign;
        t[rt<<1|1].asign*=t[rt].msign;
        t[rt<<1].sum*=t[rt].msign;
        t[rt<<1].sq*=t[rt].msign*t[rt].msign;
        t[rt<<1|1].sum*=t[rt].msign;
        t[rt<<1|1].sq*=t[rt].msign*t[rt].msign;
        t[rt].msign=1;
    }
    if(t[rt].asign)
    {
        int mid=(l+r)>>1;
        t[rt<<1].sq+=(2*t[rt].asign*t[rt<<1].sum+t[rt].asign*t[rt].asign*(mid-l+1));
        t[rt<<1].sum+=(t[rt].asign*(mid-l+1));
        t[rt<<1|1].sq+=(2*t[rt].asign*t[rt<<1|1].sum+t[rt].asign*t[rt].asign*(r-mid));
        t[rt<<1|1].sum+=(t[rt].asign*(r-mid));
        t[rt<<1].asign+=t[rt].asign;
        t[rt<<1|1].asign+=t[rt].asign;
        t[rt].asign=0;
    }
}
void range_add(int rt,int l,int r,int x,int y,ll val)//[x,y]区间加上val
{
    if(x<=l&&y>=r)
    {
        t[rt].sq=t[rt].sq+2*val*t[rt].sum+(r-l+1)*val*val;
        t[rt].sum=t[rt].sum+(r-l+1)*val;
        t[rt].asign=t[rt].asign+val;
        return ;
    }
    if(t[rt].asign!=0||t[rt].msign!=1)
        push_down(rt,l,r);
    int mid=(l+r)>>1;
    if(x<=mid)
        range_add(rt<<1,l,mid,x,y,val);
    if(y>mid)
        range_add(rt<<1|1,mid+1,r,x,y,val);
    t[rt].sum=(t[rt<<1].sum+t[rt<<1|1].sum);
    t[rt].sq=t[rt<<1].sq+t[rt<<1|1].sq;
}
void range_mul(int rt,int l,int r,int x,int y,ll val)//[x,y]区间乘上val
{
    if(x<=l&&y>=r)
    {
        t[rt].sq=val*val*t[rt].sq;
        t[rt].sum=val*t[rt].sum;
        t[rt].asign=(t[rt].asign*val);
        t[rt].msign=(t[rt].msign*val);
        return ;
    }
    if(t[rt].asign!=0||t[rt].msign!=1)
        push_down(rt,l,r);
    int mid=(l+r)>>1;
    if(x<=mid)
        range_mul(rt<<1,l,mid,x,y,val);
    if(y>mid)
        range_mul(rt<<1|1,mid+1,r,x,y,val);
    t[rt].sum=(t[rt<<1].sum+t[rt<<1|1].sum);
    t[rt].sq=t[rt<<1].sq+t[rt<<1|1].sq;
}
ll query_sum(int rt,int l,int r,int x,int y)//查询[x,y]的和
{
    if(x<=l&&y>=r)
        return t[rt].sum;
    if(t[rt].asign!=0||t[rt].msign!=1)
    push_down(rt,l,r);
    int mid=(l+r)>>1;
    ll sum=0;
    if(x<=mid)
        sum+=query_sum(rt<<1,l,mid,x,y);
    if(y>mid)
        sum+=query_sum(rt<<1|1,mid+1,r,x,y);
    return sum;
}
ll query_sq(int rt,int l,int r,int x,int y)
{
    if(x<=l&&y>=r)
        return t[rt].sq;
    if(t[rt].asign!=0||t[rt].msign!=1)
    push_down(rt,l,r);
    int mid=(l+r)>>1;
    ll sum=0;
    if(x<=mid)
        sum+=query_sq(rt<<1,l,mid,x,y);
    if(y>mid)
        sum+=query_sq(rt<<1|1,mid+1,r,x,y);
    return sum;
}
int main()
{
    cin.sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int opt,l,r;
    ll x;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        {
            cin>>a[i];
        }
    build(1,1,n);
    for(int i=1;i<=m;i++)
    {
        cin>>opt;
        if(opt==1)
        {
            cin>>l>>r;
            cout<<query_sum(1,1,n,l,r)<<'\n';
        }
        if(opt==2)
        {
            cin>>l>>r;
            cout<<query_sq(1,1,n,l,r)<<'\n';
        }
        if(opt==3)
        {
            cin>>l>>r>>x;
            range_mul(1,1,n,l,r,x);
        }
        if(opt==4)
        {
            cin>>l>>r>>x;
            range_add(1,1,n,l,r,x);
        }
    }
    return 0;
}

//离散化模板
vector<int> v;
int scatter(vector<int> &v)
{
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    return v.size();
}
inline int getid(int x)
{
    return lower_bound(v.begin(), v.end(), x) - v.begin() + 1;
}


//离散化板子  **未验证
void scatter(int a[],int n)
{
    for(int i=0;i<n;i++)
    {
        b[i]=a[i];
    }
    sort(b,b+n);
    int sz=unique(b,b+n)-b;
    for(int i=0;i<n;i++)
    {
        c[i]=lower_bound(b,b+sz,a[i])-b;
    }
}

#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>
#include<stack>
#include<map>
#include<sstream>
using namespace std;
typedef long long ll;
const int maxn = 1e3 + 10;
const int INF = 1 << 30;
int T, n, m;
struct edge
{
    int from, to, dist;
    edge(int u, int v, int d):from(u), to(v), dist(d){}
    edge(){}
};
struct Heapnode
{
    int d, u;//d为距离，u为起点
    Heapnode(){}
    Heapnode(int d, int u):d(d), u(u){}
    bool operator <(const Heapnode & a)const
    {
        return d > a.d;//这样优先队列先取出d小的
    }
};
struct Dijkstra
{
    int n, m;
    vector<edge>edges;//存边的信息
    vector<int>G[maxn];//G[i]表示起点为i的边的序号集
    bool v[maxn];//标记点是否加入集合
    int d[maxn];//起点s到各个点的最短路
    int p[maxn];//倒叙记录路径
    Dijkstra(){}
    void init(int n)
    {
        this -> n = n;
        for(int i = 0; i < n; i++)G[i].clear();
        edges.clear();
    }
    void addedge(int from, int to, int dist)
    {
        edges.push_back(edge(from, to, dist));
        m = edges.size();
        G[from].push_back(m - 1);//存以from为起点的下一条边
    }
    void dijkstra(int s)//以s为起点
    {
        priority_queue<Heapnode>q;
        for(int i = 0; i < n; i++)d[i] = INF;
        d[s] = 0;
        memset(v, 0, sizeof(v));
        memset(p, -1, sizeof(p));
        q.push(Heapnode(0, s));
        while(!q.empty())
        {
            Heapnode now = q.top();
            q.pop();
            int u = now.u;//当前起点
            if(v[u])continue;//如果已经加入集合，continue
            v[u] = 1;
            for(int i = 0; i < G[u].size(); i++)
            {
                edge& e = edges[G[u][i]];//引用节省代码
                if(d[e.to] > d[u] + e.dist)
                {
                    d[e.to] = d[u] + e.dist;
                    p[e.to] = G[u][i];//记录e.to前的边的编号，p存的是边的下标,这样可以通过边找出之前的点以及每条路的路径，如果用邻接矩阵存储的话这里可以直接存节点u
                    q.push(Heapnode(d[e.to], e.to));
                }
            }
        }
    }
    void output(int u)
    {
        for(int i = 0; i < n; i++)
        {
            if(i == u)continue;
            printf("从%d到%d距离是：%2d   ", u, i, d[i]);
            stack<int>q;//存的是边的编号
            int x = i;//x就是路径上所有的点
            while(p[x] != -1)
            {
                q.push(x);
                x = edges[p[x]].from;//x变成这条边的起点
            }
            cout<<u;
            while(!q.empty())
            {
                cout<<"->"<<q.top();
                q.pop();
            }
            cout<<endl;
        }
    }
};
Dijkstra ans;
int main()
{
    cin >> n >> m;
    ans.init(n);
    for(int i = 0; i < m; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        ans.addedge(u, v, w);
    }
    int u = 0;
    ans.dijkstra(u);
    ans.output(u);
}

//组合数前缀和(1e9版本)
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;
const int N = 700005;
typedef long long ll;
const ll mod = 998244353ll;
ll fac[N], rfac[N];
ll ksm(ll x, ll k)
{
    ll s = 1;
    while (k)
    {
        if (k & 1)
            s = s * x % mod;
        x = x * x % mod;
        k >>= 1;
    }
    return s;
}
ll G = 3;
struct NTT
{
    ll ft[N];
    int rev[N];
    void init(int n)
    {
        int k;
        for (k = 0; (1 << k) < n; k++)
            ;
        for (int i = 0; i < n; i++) rev[i] = rev[i >> 1] >> 1 | ((i & 1) << (k - 1));
    }
    void trans(ll *a, int n, int ty)
    {
        for (int i = 0; i < n; i++)
            if (i < rev[i])
                swap(a[i], a[rev[i]]);
        ft[0] = 1;
        for (int m = 1; m < n; m <<= 1)
        {
            ll t0 = ksm(G, mod - 1 + ty * (mod - 1) / (m << 1));
            for (int i = 1; i < m; i++) ft[i] = ft[i - 1] * t0 % mod;
            for (int k = 0; k < n; k += (m << 1))
                for (int i = k; i < k + m; i++)
                {
                    ll t0 = a[i], t1 = a[i + m] * ft[i - k] % mod;
                    a[i] = (t0 + t1) % mod;
                    a[i + m] = (t0 - t1 + mod) % mod;
                }
        }
    }
    void dft(ll *a, int n)
    {
        trans(a, n, 1);
    }
    void idft(ll *a, int n)
    {
        trans(a, n, -1);
        ll t0 = ksm(n, mod - 2);
        for (int i = 0; i < n; i++) a[i] = a[i] * t0 % mod;
    }
} ntt;
ll A[N], B[N], C[N], ff[N];
ll inv(ll x)
{
    return ksm(x, mod - 2);
}
void calc(ll *st, ll *ed, int d, int k)
{
    ff[0] = 1;
    for (int i = 0; i <= d; i++) ff[0] = ff[0] * (k - i) % mod;
    for (int i = 1; i <= d; i++) ff[i] = ff[i - 1] * (i + k) % mod * inv(i + k - d - 1) % mod;

    int len;
    for (len = 1; len <= 3 * d; len <<= 1)
        ;
    ntt.init(len);
    for (int i = 0; i <= d; i++)
    {
        A[i] = st[i] * rfac[i] % mod * rfac[d - i] % mod;
        if ((d - i) & 1)
            A[i] = mod - A[i];
    }
    for (int i = 0; i <= 2 * d; i++) B[i] = inv(i - d + k);
    ntt.dft(A, len);
    ntt.dft(B, len);
    for (int i = 0; i < len; i++) C[i] = A[i] * B[i] % mod;
    ntt.idft(C, len);
    for (int i = 0; i <= d; i++)
    {
        ed[i] = C[i + d] * ff[i] % mod;
    }
    for (int i = 0; i < len; i++) A[i] = B[i] = C[i] = 0;
}
ll qz[N], hz[N], V, rV;
ll ag[N], revg[N], dg[N];
ll af[N], df[N];
int n;
void work(int r, vector<ll> &g, vector<ll> &f)
{
    if (!r)
    {
        g.push_back(1);
        f.push_back(0);
        return;
    }
    if (r & 1)
    {
        work(r - 1, g, f);
        for (int i = 0; i < r; i++) ag[i] = g[i];
        calc(ag, revg, r - 1, (-n - 1) * rV % mod);
        for (int i = 0; i < r; i++) g[i] = g[i] * (i * V % mod + r) % mod;
        ll p = 1;
        for (int i = 1; i <= r; i++) p = p * (r * V + i) % mod;
        g.push_back(p);
        for (int i = 0; i < r; i++)
        {
            if ((r - 1) & 1)
                f[i] = (f[i] - revg[i]) % mod;
            else
                f[i] = (f[i] + revg[i]) % mod;
            f[i] = f[i] * (i * V + r) % mod;
        }
        hz[r + 1] = 1;
        for (int i = r; i >= 0; i--) hz[i] = hz[i + 1] * (i + r * V) % mod;
        qz[-1] = 1;
        for (int i = 0; i <= r; i++) qz[i] = qz[i - 1] * (n - i - r * V) % mod;
        ll sum = 0;
        for (int i = 0; i < r; i++) sum = (sum + hz[i + 1] * qz[i - 1] % mod) % mod;
        f.push_back(sum);
        return;
    }

    int d = r >> 1;
    work(d, g, f);
    for (int i = 0; i <= d; i++) ag[i] = g[i];
    calc(ag, ag + d + 1, d, d + 1);
    calc(ag, dg, 2 * d, d * rV % mod);
    calc(ag, revg, 2 * d, (-n - 1) * rV % mod);
    for (int i = 0; i <= d; i++) af[i] = f[i];
    calc(af, af + d + 1, d, d + 1);
    calc(af, df, 2 * d, d * rV % mod);
    f.resize(r + 1);
    for (int i = 0; i <= r; i++)
    {
        ll s = df[i] * revg[i] % mod;
        if (d & 1)
            s = mod - s;
        f[i] = (s + af[i] * dg[i] % mod) % mod;
    }
    g.resize(r + 1);
    for (int i = 0; i <= r; i++) g[i] = ag[i] * dg[i] % mod;
}
void sol()
{
    int m;
    scanf("%d%d", &n, &m);
    V = sqrt(n);
    rV = inv(V);
    vector<ll> g, f;
    work(V, g, f);
    ll fac = 1;
    for (int i = 0; i < V; i++) fac = fac * g[i] % mod;
    for (int i = V * V + 1; i <= n; i++) fac = fac * i % mod;
    for (int i = 0; i <= V; i++) ag[i] = g[i];
    calc(ag, revg, V, (-n - 1) * rV % mod);
    qz[-1] = 1;
    for (int i = 0; i <= V; i++) qz[i] = qz[i - 1] * revg[i] % mod;
    hz[V] = 1;
    for (int i = V * V + 1; i <= n; i++) hz[V] = hz[V] * i % mod;
    for (int i = V - 1; i >= 0; i--) hz[i] = hz[i + 1] * g[i] % mod;
    ll sum = 0;
    int cr = 0;
    for (int i = 0; i < V && (i + 1) * V - 1 <= m; i++)
    {
        ll t = hz[i + 1] * qz[i - 1] % mod * f[i] % mod;
        if ((i * V) & 1)
            t = mod - t;
        sum = (sum + t) % mod;
        cr = i + 1;
    }
    sum = sum * inv(fac) % mod;
    ll C = fac, G = 1;
    for (int j = 0; j < cr; j++) G = G * g[j] % mod;
    int s = n - cr * V, cp = 0;
    for (int i = 0; i < V && (i + 1) * V <= s; i++) G = G * g[i] % mod, cp = i + 1;
    for (int i = cp * V + 1; i <= s; i++) G = G * i % mod;
    C = C * inv(G) % mod;
    for (int i = cr * V; i <= m; i++)
    {
        if (i != cr * V)
        {
            C = C * inv(i) % mod * (n - i + 1) % mod;
        }
        sum = (sum + C) % mod;
    }
    sum = (sum % mod + mod) % mod;
    printf("%lld\n", sum);
}
int main()
{
    fac[0] = 1;
    for (int i = 1; i < N; i++) fac[i] = fac[i - 1] * i % mod;
    rfac[N - 1] = inv(fac[N - 1]);
    for (int i = N - 2; i >= 0; i--) rfac[i] = rfac[i + 1] * (i + 1) % mod;
    int T;
    scanf("%d", &T);
    while (T--) sol();

    return 0;
}
