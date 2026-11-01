#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
struct F{int h,r,o,t;}; struct E{int t,e,f,r;};
struct R{std::vector<int>b;std::array<int,4>g;bool c;double conf;};
struct W{double t,l;int n;std::array<int,3>f,r;};
static std::vector<F>fs;static std::vector<std::unordered_map<int,std::vector<E>>>gr;static std::unordered_map<int,std::vector<R>>rs;static int nr,oldest;static std::vector<unsigned char>first_identical_fact;static std::vector<int>canonical_fact_id;
static long I(PyObject*x){return PyLong_AsLong(x);}
static const std::vector<E>empty;
static const std::vector<E>&edges(int r,int e){auto it=gr[r].find(e);return it==gr[r].end()?empty:it->second;}
static int bound(const std::vector<E>&es,int t){return std::lower_bound(es.begin(),es.end(),t,[](const E&e,int t){return e.t<t;})-es.begin();}
static bool ok(const R&r,const std::array<int,4>&ns,int n){if(!r.c)return true;for(int i=0;i<n;i++)for(int j=0;j<i;j++)if(r.g[i]>=0&&r.g[i]==r.g[j]&&ns[i]!=ns[j])return false;return true;}
static auto ft(int fid,int r){const F&f=fs[fid];return r<nr?std::make_tuple(f.h,r,f.o,f.t):std::make_tuple(f.o,r,f.h,f.t);}
static bool prefer(const W&a,const W&b){if(a.n!=b.n)return a.n<b.n;int at=fs[a.f[a.n-1]].t,bt=fs[b.f[b.n-1]].t;if(at!=bt)return at>bt;for(int i=0;i<a.n;i++){auto x=ft(a.f[i],a.r[i]),y=ft(b.f[i],b.r[i]);if(x!=y)return x<y;}return false;}
static PyObject*init(PyObject*,PyObject*args){PyObject*facts,*rules;int n;if(!PyArg_ParseTuple(args,"OOi",&facts,&rules,&n))return nullptr;nr=n;fs.clear();gr.clear();gr.resize(2*n);rs.clear();oldest=2147483647;
for(Py_ssize_t i=0;i<PySequence_Size(facts);i++){PyObject*x=PySequence_GetItem(facts,i);F f{(int)I(PySequence_Fast_GET_ITEM(x,0)),(int)I(PySequence_Fast_GET_ITEM(x,1)),(int)I(PySequence_Fast_GET_ITEM(x,2)),(int)I(PySequence_Fast_GET_ITEM(x,3))};Py_DECREF(x);fs.push_back(f);oldest=std::min(oldest,f.t);gr[f.r][f.h].push_back({f.t,f.o,(int)i,f.r});gr[f.r+n][f.o].push_back({f.t,f.h,(int)i,f.r+n});}
for(auto&g:gr)for(auto&x:g)std::sort(x.second.begin(),x.second.end(),[](const E&a,const E&b){return std::tie(a.t,a.e,a.f)<std::tie(b.t,b.e,b.f);});
PyObject*k,*a;Py_ssize_t pos=0;while(PyDict_Next(rules,&pos,&k,&a)){auto&v=rs[std::stoi(PyUnicode_AsUTF8(k))];for(Py_ssize_t i=0;i<PyList_Size(a);i++){PyObject*d=PyList_GetItem(a,i);R r;r.g.fill(-1);r.conf=PyFloat_AsDouble(PyDict_GetItemString(d,"conf"));PyObject*b=PyDict_GetItemString(d,"body_rels");for(Py_ssize_t j=0;j<PyList_Size(b);j++)r.b.push_back(I(PyList_GetItem(b,j)));PyObject*c=PyDict_GetItemString(d,"var_constraints");r.c=c&&PyList_Size(c)>0;if(c)for(Py_ssize_t j=0;j<PyList_Size(c);j++){PyObject*z=PyList_GetItem(c,j);for(Py_ssize_t u=0;u<PyList_Size(z);u++)r.g[I(PyList_GetItem(z,u))]=j;}if(r.b.size()>=1&&r.b.size()<=3)v.push_back(r);}}
// Precompute exact quadruple identity once, rather than deduplicating a
// potentially large incident neighborhood for every forecasting query.
std::map<std::tuple<int,int,int,int>,int> first_index;
first_identical_fact.resize(fs.size());
canonical_fact_id.resize(fs.size());
for(size_t i=0;i<fs.size();i++){
const F&f=fs[i];auto key=std::make_tuple(f.h,f.r,f.o,f.t);
auto ret=first_index.emplace(key,(int)i);
first_identical_fact[i]=ret.second?1:0;
canonical_fact_id[i]=ret.first->second;
}
Py_RETURN_NONE;}
static PyObject*query(PyObject*,PyObject*args){int known,rel,role,qt;if(!PyArg_ParseTuple(args,"iiii",&known,&rel,&role,&qt))return nullptr;int hr=rel+(role?nr:0);double D=qt-oldest;auto rec=[&](int f){return D>0?std::exp(-(qt-fs[f].t)/D):1.;};
std::map<int,std::vector<W>>out;
// Store the recency/logical Pareto frontier per endpoint. This is sufficient for
// every nonnegative weight and avoids enumerating timestamp combinations.
auto record=[&](int e,W w){auto&v=out[e];for(auto&i:v)if(i.t>=w.t&&i.l>=w.l){if(i.t!=w.t||i.l!=w.l||!prefer(w,i))return;}
v.erase(std::remove_if(v.begin(),v.end(),[&](const W&i){return w.t>=i.t&&w.l>=i.l;}),v.end());v.push_back(w);};
for(int r=role?nr:0;r<(role?2*nr:nr);r++){const auto&es=edges(r,known);int hi=bound(es,qt);for(int i=0;i<hi;i++)record(es[i].e,W{rec(es[i].f),0.,1,{es[i].f,-1,-1},{r,-1,-1}});}
auto rit=rs.find(hr);if(rit!=rs.end())for(const R&r:rit->second){int n=r.b.size();const auto&a=edges(r.b[0],known);int hi=bound(a,qt);std::map<int,std::vector<E>>first;
for(int i=0;i<hi;i++){std::array<int,4>ns{known,a[i].e,-1,-1};if(!ok(r,ns,2))continue;if(n==1)record(a[i].e,W{rec(a[i].f),r.conf,1,{a[i].f,-1,-1},{r.b[0],-1,-1}});else first[a[i].e].push_back(a[i]);}
if(n==1)continue;
for(const auto&kv:first){int n1=kv.first;const auto&one=kv.second;const auto&b=edges(r.b[1],n1);int bh=bound(b,qt);size_t cursor=0;int best=-1;
struct P{int t,f1,f2;double sum;};std::map<int,std::vector<P>>prefix;
for(int j=0;j<bh;j++){while(cursor<one.size()&&one[cursor].t<=b[j].t){int f=one[cursor++].f;if(best<0||rec(f)>rec(best)||(rec(f)==rec(best)&&ft(f,r.b[0])<ft(best,r.b[0])))best=f;}
if(best<0)continue;std::array<int,4>ns{known,n1,b[j].e,-1};if(!ok(r,ns,3))continue;double sum=rec(best)+rec(b[j].f);
if(n==2)record(b[j].e,W{sum/2,r.conf,2,{best,b[j].f,-1},{r.b[0],r.b[1],-1}});else prefix[b[j].e].push_back({b[j].t,best,b[j].f,sum});}
if(n==2)continue;
for(const auto&pv:prefix){int n2=pv.first;const auto&two=pv.second;const auto&c=edges(r.b[2],n2);int ch=bound(c,qt);size_t u=0;int bp=-1;
for(int z=0;z<ch;z++){while(u<two.size()&&two[u].t<=c[z].t){int cur=u++;if(bp<0||two[cur].sum>two[bp].sum)bp=cur;else if(two[cur].sum==two[bp].sum){W x{0,0,3,{two[cur].f1,two[cur].f2,c[z].f},{r.b[0],r.b[1],r.b[2]}},y{0,0,3,{two[bp].f1,two[bp].f2,c[z].f},{r.b[0],r.b[1],r.b[2]}};if(prefer(x,y))bp=cur;}}
if(bp<0)continue;std::array<int,4>ns{known,n1,n2,c[z].e};if(!ok(r,ns,4))continue;const P&p=two[bp];record(c[z].e,W{(p.sum+rec(c[z].f))/3,r.conf,3,{p.f1,p.f2,c[z].f},{r.b[0],r.b[1],r.b[2]}});}
}
}
}
PyObject*res=PyDict_New();for(auto&kv:out){PyObject*v=PyList_New(kv.second.size());for(size_t j=0;j<kv.second.size();j++){const W&w=kv.second[j];PyObject*path=PyList_New(w.n);for(int k=0;k<w.n;k++)PyList_SET_ITEM(path,k,Py_BuildValue("ii",w.f[k],w.r[k]));PyList_SET_ITEM(v,j,Py_BuildValue("ddN",w.t,w.l,path));}PyObject*k=PyLong_FromLong(kv.first);PyDict_SetItem(res,k,v);Py_DECREF(k);Py_DECREF(v);}return res;}
// Strict-past one-hop endpoint recurrence under the current query direction.
// Counts all visible incident facts, regardless of their base relation; no future/current facts.
static PyObject*recurrence(PyObject*,PyObject*args){
int known,role,qt,unique_fact=0;
if(!PyArg_ParseTuple(args,"iii|i",&known,&role,&qt,&unique_fact))return nullptr;
PyObject*res=PyDict_New();
for(int r=role?nr:0;r<(role?2*nr:nr);r++){
const auto&es=edges(r,known);int hi=bound(es,qt);
for(int i=0;i<hi;i++){
if(unique_fact&&!first_identical_fact[es[i].f])continue;
PyObject*key=PyLong_FromLong(es[i].e);
PyObject*prev=PyDict_GetItemWithError(res,key);
if(!prev&&PyErr_Occurred()){Py_DECREF(key);Py_DECREF(res);return nullptr;}
long n=prev?PyLong_AsLong(prev):0;
PyObject*v=PyLong_FromLong(n+1);
PyDict_SetItem(res,key,v);
Py_DECREF(v);Py_DECREF(key);
}}
return res;
}
static PyMethodDef methods[]={{"init",init,METH_VARARGS,"Load observed facts and complete temporal rules."},{"query",query,METH_VARARGS,"Return exact nonnegative-score endpoint frontiers."},{"recurrence",recurrence,METH_VARARGS,"Count strict-past one-hop endpoints for the known entity and direction."},{nullptr,nullptr,0,nullptr}};static PyModuleDef module={PyModuleDef_HEAD_INIT,"grounded_chain_native",nullptr,-1,methods};PyMODINIT_FUNC PyInit_grounded_chain_native(){return PyModule_Create(&module);}
