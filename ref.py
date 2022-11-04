# Barak Morris
import math
import sympy
import numpy as np

s,t = sympy.symbols('s t')

# The bisection method is good for finding roots of
# continuous functions where one knows two values of
# opposite signs
def bisection(a,b,tolerance,maxiterations,function):
    i = 1
    fa = function(a)
    while(i <= maxiterations):
        p = a + (b-a)/2
        fp = function(p)
        if(fp == 0 or (b-a)/2 < tolerance):
            print("solution found: (" + str(p)+ ") after " + str(i) + " iterations")
            return(p,i)
        i = i + 1
        if(fa*fp > 0):
            a = p
            fa = fp
        else:
            b = p
    print("method failed after " + str(maxiterations) + " iterations")
    return(p,i)

# A fixed point has the property where g(p) = p
# where g is a continous function
# f(x) = x - g(x) has roots at p
def fixedpoint(p0,tolerance,maxiterations,function):
    i = 1
    while(i <= maxiterations):
        p = function(p0)
        if(abs(p-p0) < tolerance):
            print("solution found: (" + str(p)+ ") after " + str(i) + " iterations")
            return(p,i)
        i = i + 1
        p0 = p
    print("method failed after " + str(maxiterations) + " iterations")
    return(p,i)

# Newton's Method
# Finding root with initial approximation
def newton(p0,tol,maxit,f,fp,verbose):
    i = 1
    if verbose: print(p0)
    while(i <= maxit):
        p = p0 - f(p0) / fp(p0)
        if verbose: print(p)
        if(abs(p-p0) < tol):
            return(p,i)
        i = i + 1
        p0 = p
    return(None,i)

# Modified Newton's Method
# Finding root with initial approximation
def modifiednewton(p0,tol,maxit,f,fp,fpp,verbose):
    i = 1
    if verbose: print(p0)
    while(i <= maxit):
        p = p0 - (f(p0) * fp(p0))/(fp(p0)**2-f(p0)*fpp(p0))
        if verbose: print(p)
        if(abs(p-p0) < tol):
            return(p,i)
        i = i + 1
        p0 = p
    return(None,i)

# Secant Method
# Finding roots with two initial approximations
def secant(p0,p1,tol,maxit,f,verbose):
    i = 2
    q0 = f(p0)
    q1 = f(p1)
    if verbose: print(p0,"\n",p1)
    while(i <= maxit):
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if verbose: print(p)
        if(abs(p-p1) < tol):
            return(p,i)
        i = i + 1
        p0 = p1
        q0 = q1
        p1 = p
        q1 = f(p)
    return(None,i)

# False Position
# Finding roots with two opposite signs
def falseposition(p0,p1,tol,maxit,f,verbose):
    i = 2
    q0 = f(p0)
    q1 = f(p1)
    if verbose: print(p0,"\n",p1)
    while(i <= maxit):
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if verbose: print(p)
        if(abs(p-p1) < tol):
            return(p,i)
        i = i + 1
        q = f(p)
        if(q*q1 < 0):
            p0 = p1
            q0 = q1
        p1 = p
        q1 = q
    return(None,i)

# Muller's Method
# Find a root given three approximations
def muller(p0,p1,p2,tol,maxit,f):
    h1 = p1-p0
    h2 = p2-p1
    s1 = (f(p1)-f(p0))/h1
    s2 = (f(p2)-f(p1))/h2
    d  = (s2-s1)/(h2+h1)
    i  = 3
    while(i <= maxit):
        b = s2+h2*d
        D = math.sqrt(b**2-4*f(p2)*d)
        if(abs(b-D)<abs(b+D)):
            E = b+D
        else:
            E = b-D
        h = -2*f(p2)/E
        p = p2+h
        if(abs(h)<tol):
            p = p.real - p.imag*1j
            print("Found x=" + str(p) + " after " + str(i) + " iterations")
            return(p,i)
        p0 = p1
        p1 = p2
        p2 = p
        h1 = p1-p0
        h2 = p2-p1
        s1 = (f(p1)-f(p0))/h1
        s2 = (f(p2)-f(p1))/h2
        d  = (s2-s1)/(h2+h1)
        i  = i+1
    print("Method failed after " + str(maxit) + " iterations")
    return(None,i)

# Linear
def linear(seq,n):
    i = 1
    while(i <= n):
        print(seq(i))
        i = i + 1
    return

# Aitken
# Improve the convergence of a sequence
def aitken(seq,n):
    i = 1
    while(i <= n):
        p = seq(i) - (seq(i+1) - seq(i))**2/(seq(i+2) - 2*seq(i+1) + seq(i))
        print(p)
        i = i + 1
    return

# Steffensen
# Finding a solution to p = g(p) with initial approximation
def steffensen(p0,tol,maxit,g,verbose):
    i = 1
    if(verbose): print(p0)
    while(i <= maxit):
        p1 = g(p0)
        p2 = g(p1)
        p = p0 - (p1 - p0)**2 / (p2 - 2*p1 + p0)
        if(verbose): print(p)
        if(abs(p-p0) < tol):
            return(p,i)
        i = i + 1
        p0 = p
    return(p,i)

# Evaluate the hermite interpolation function
# Given the q list from the hermite function
def evalpolyhermite(q,xn):
    x = sympy.symbols('x')
    i = 1
    n = 0
    h = q[0]
    expr = 1
    while(i < len(q)):
        expr = sympy.Mul(expr,(x-xn[n]))
        h = sympy.Add(h,expr*q[i])
        if(i%2==0): 
            n = n + 1
        i = i + 1
    h = sympy.simplify(h)
    return h

# Hermite interpolation
# Input the x values, the function and the derivative values
def hermite(x,fx,fpx):
    n = len(x) - 1
    Q = np.zeros((2*n+2,2*n+2))
    z = np.zeros(2*n+2)
    i = 0
    while(i <= n):
        z[2*i] = x[i]
        z[2*i+1] = x[i]
        Q[2*i,0] = fx[i]
        Q[2*i+1,0] = fx[i]
        Q[2*i+1,1] = fpx[i]
        if(i != 0):
            Q[2*i,1] = (Q[2*i,0] - Q[2*i-1,0])/(z[2*i]-z[2*i-1])
        i = i + 1
    i = 2
    while(i <= (2*n+1)):
        j = 2
        while(j <= i):
            Q[i,j] = (Q[i,j-1] - Q[i-1,j-1])/(z[i]-z[i-j])
            j = j + 1
        i = i + 1
    return(np.diag(Q))

# Cubic spline interpolation
# Given the x values and the function values
def cubicspline(x,fx):
    n = len(x) - 1
    h = np.zeros(n+1)
    a = np.zeros(n+1)
    l = np.zeros(n+1)
    u = np.zeros(n+1)
    z = np.zeros(n+1)
    c = np.zeros(n+1)
    b = np.zeros(n+1)
    d = np.zeros(n+1)
    i = 0
    while(i <= n-1):
        h[i] = x[i+1] - x[i]
        i = i + 1
    i = 1
    while(i <= n-1):
        a[i] = (3*h[i]**-1)*(fx[i+1]-fx[i])-(3*h[i-1]**-1)*(fx[i]-fx[i-1])
        i = i + 1
    l[0] = 1
    i = 1
    while(i <= n-1):
        l[i] = 2*(x[i+1]-x[i-1])-h[i-1]*u[i-1]
        u[i] = h[i]/l[i]
        z[i] = (a[i]-h[i-1]*z[i-1])/l[i]
        i = i + 1
    l[n] = 1
    j = n - 1
    while(j >= 0):
        c[j] = z[j]-u[j]*c[j+1]
        b[j] = (fx[j+1]-fx[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
        j = j - 1
    return(fx,b,c,d)

# Clamped cubic spline interpolation
# Given the x values and the function values
def clampedcubicspline(x,fx,fpx):
    n = len(x) - 1
    i = 0
    a = np.zeros(n+1)
    h = np.zeros(n+1)
    b = np.zeros(n+1)
    c = np.zeros(n+1)
    d = np.zeros(n+1)
    l = np.zeros(n+1)
    u = np.zeros(n+1)
    z = np.zeros(n+1)
    while(i <= (n-1)):
        h[i] = x[i+1]-x[i]
        i = i + 1
    a[0] = 3*(fx[1]-fx[0])/h[0]-3*fpx[0]
    a[n] = 3*fpx[n]-3*(fx[n]-fx[n-1])/h[n-1]
    i = 1
    while(i <= (n-1)):
        a[i] = (3/h[i])*(fx[i+1]-fx[i])-(3/h[i-1])*(fx[i]-fx[i-1])
        i = i + 1
    l[0] = 2*h[0]
    u[0] = .5
    z[0] = a[0]/l[0]
    i = 1
    while(i <= (n-1)):
        l[i] = 2*(x[i+1]-x[i-1])-h[i-1]*u[i-1]
        u[i] = h[i]/l[i]
        z[i] = (a[i]-h[i-1]*z[i-1])/l[i]
        i = i + 1
    l[n] = h[n-1]*(2-u[n-1])
    z[n] = (a[n]-h[n-1]*z[n-1])/l[n]
    c[n] = z[n]
    j = n-1
    while(j >= 0):
        c[j] = z[j] - u[j]*c[j+1]
        b[j] = (fx[j+1]-fx[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
        j = j - 1
    return(fx,b,c,d)

# Creates a composite value estimation
def composite(x0,xn,n,method,fun):
    if(method.__name__ != 'trapezoidal'):n=n/2
    j = 0
    val = 0
    h = (xn-x0)/n
    while(j < n):
        val += method(x0+j*h,x0+(j+1)*h,fun)
        j += 1
    return val

# Trapezoidal method
def trapezoidal(x0,x1,f):
    h = x1 - x0
    return((h/2)*(f.evalf(subs={s:x0})+f.evalf(subs={s:x1})))

# Simpson's method
def simpson(x0,x2,f):
    h = (x2-x0)/2
    x1 = x0 + h
    return((h/3)*(f.evalf(subs={s:x0})+4*f.evalf(subs={s:x1})+f.evalf(subs={s:x2})))

# Midpoint method
def midpoint(xm1,x1,f):
    h = (x1-xm1)/2
    mp = xm1 + h
    return(2*h*f.evalf(subs={s:mp}))

# Simpson's three eighths method
def simpsonte(x0,x3,f):
    h = (x3-x0)/3
    x1 = x0 + h
    x2 = x1 + h
    return((3*h/8)*(f.evalf(subs={s:x0})+3*f.evalf(subs={s:x1})+3*f.evalf(subs={s:x2})+f.evalf(subs={s:x3})))

# Newton's method for sympy expr
def newtonsympy(p0,tol,maxit,f,fp,verbose):
    i = 1
    if verbose: print(p0)
    while(i <= maxit):
        p = p0 - f.evalf(subs={s:p0}) / fp.evalf(subs={s:p0})
        if verbose: print(p)
        if(abs(p-p0) < tol):
            return(p,i)
        i = i + 1
        p0 = p
    return(p,i)

# This allows you to table off a sequence
def methodtabler(iterations,method):
    p = 1
    n = 0
    while (n < iterations):
        p = method(p)
        n = n + 1
        print("n: " + str(n) + " p: " + str(p))
    return

# Euler's method
def eulers(a,b,N,alpha,fty):
    h = (b-a)/N
    t = []
    w = []
    i = 1
    t.append(a)
    w.append(alpha)
    while(i<=N):
        w.append(w[i-1] + h*fty(t[i-1],w[i-1]))
        t.append(round(a + i*h,3))
        i = i + 1
    return(t,w)

# Taylor's 2nd Order Method
def taylors2(a,b,N,alpha,fty,ftyp):
    h = (b-a)/N
    t = []
    w = []
    i = 1
    t.append(a)
    w.append(alpha)
    while(i<=N):
        w.append(w[i-1] + h*fty(t[i-1],w[i-1]) + (h**2/2)*ftyp(t[i-1],w[i-1]))
        t.append(round(a + i*h,3))
        i = i + 1
    return(t,w)

# Taylor's Fourth Order Method
def taylors4(a,b,N,alpha,fty,ftyp,ftypp,ftyppp):
    h = (b-a)/N
    t = []
    w = []
    i = 1
    t.append(a)
    w.append(alpha)
    while(i<=N):
        w.append(w[i-1] + h*fty(t[i-1],w[i-1]) + (h**2/2)*ftyp(t[i-1],w[i-1]) + (h**3/6)*ftypp(t[i-1],w[i-1]) + (h**4/24)*ftyppp(t[i-1],w[i-1]))
        t.append(round(a + i*h,3))
        i = i + 1
    return(t,w)

# Evaluates the lagrange polynomial at a given x0
def lagrange(xn,fxn,x0):
    product = 1
    tot = 0
    for k in range(len(xn)):
        product = 1
        for i in range(len(xn)):
            if(i!=k):
                product = product*(x0-xn[i])/(xn[k]-xn[i])
        tot = tot + fxn[k]*product
    return tot

# Evaluates a function at each x
def evaluate(fcn,x):
    y = []
    for i in x:
        y.append(fcn(i))
    return y

# Finds the error between an estimation and a function at each t
def finderror(t,y,fcn):
    n = len(t)
    error = []
    i = 0
    while(i<n):
        error.append(abs(y[i]-fcn(t[i])))
        i = i + 1
    return error

# This prints the runge kutta fifth order approximation
def rk5(a,b,alpha,tol,hmax,hmin,f,sol):
    t = a
    w = alpha
    h = hmax
    flag = 1
    error = abs(sol(t)-w)
    print("t: ",t," w: ",w," error: ",error)
    while(flag == 1):
        k1 = h*f(t,w)
        k2 = h*f(t+(1/4)*h,w+(1/4)*k1)
        k3 = h*f(t+(3/8)*h,w+(3/32)*k1+(9/32)*k2)
        k4 = h*f(t+(12/13)*h,w+(1932/2197)*k1-(7200/2197)*k2+(7296/2197)*k3)
        k5 = h*f(t+h,w+(439/216)*k1-8*k2+(3680/513)*k3-(845/4104)*k4)
        k6 = h*f(t+(1/2)*h,w-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5)
        r = (1/h)*abs((1/360)*k1-(128/4275)*k3-(2197/75240)*k4+(1/50)*k5+(2/55)*k6)
        if(r<=tol):
            t = t+h
            w = w+(25/216)*k1+(1408/2565)*k3+(2197/4104)*k4-(1/5)*k5
            error = abs(sol(t)-w)
            print("t: ",t," w: ",w," h: ",h," error: ",error)
        delta = 0.84*(tol/r)**(1/4)
        if(delta<=0.1):
            h = 0.1*h
        elif(delta>=4):
            h = 4*h
        else:
            h = delta*h
        if(h>hmax):
            h = hmax
        if(t>=b):
            flag = 0
        elif(t+h>b):
            h = b-t
        elif(h<hmin):
            flag = 0
            print("minimum h exceeded")
            return
    return

# Modified euler's method
def modifiedeulers(a,b,N,alpha,fty,sol):
    verbose = False
    if sol != None:
        verbose = True
    h = (b-a)/N
    t = []
    w = []
    i = 1
    t.append(a)
    w.append(alpha)
    if verbose: print("n: 1"," w: ",w[0]," y: ",sol(a)," error: 0.0")
    while(i<=N):
        t.append(a + i*h)
        w.append(w[i-1] + (h/2)*(fty(t[i],w[i-1])+fty(t[i-1],w[i-1]+h*fty(t[i-1],w[i-1]))))
        error = abs(sol(t[i])-w[i])
        if verbose: print("n: ",i," w: ",w[i]," y: ",sol(t[i])," error: ",error)
        i = i + 1
    return(t,w)

# Midpoint method / runge kutta order two
def rk2(a,b,N,alpha,fty,sol):
    verbose = False
    if sol != None:
        verbose = True
    h = (b-a)/N
    t = []
    w = []
    i = 1
    t.append(a)
    w.append(alpha)
    if verbose: print("n: 1"," w: ",w[0]," y: ",sol(a)," error: 0.0")
    while(i<=N):
        t.append(a + i*h)
        w.append(w[i-1] + h*fty(t[i-1]+h/2,w[i-1]+(h/2)*fty(t[i-1],w[i-1])))
        error = abs(sol(t[i])-w[i])
        if verbose: print("n: ",i," w: ",w[i]," y: ",sol(t[i])," error: ",error)
        i = i + 1
    return(t,w)

# Runge kutta fourth order approximation for a system
def rk4sys(a,b,N,alpham,fjtu):
    m = len(alpham)
    h = (b-a)/N
    t = a
    values,k1,k2,k3,k4,wd,w = [],[],[],[],[],[],[],[]
    for i in range(m):
        k1.append(0);k2.append(0);k3.append(0);k4.append(0);wd.append(0);error.append(0);w.append(alpham[i]);values.append(0)
    ti=[a]
    xi=[w]
    for i in range(1,N+1):
        for j in range(m): wd[j] = w[j]
        for j in range(m): k1[j] = h*fjtu[j](t,*wd)
        for j in range(m): wd[j] = w[j] + k1[j]/2
        for j in range(m): k2[j] = h*fjtu[j](t+h/2,*wd)
        for j in range(m): wd[j] = w[j] + k2[j]/2
        for j in range(m): k3[j] = h*fjtu[j](t+h/2,*wd)
        for j in range(m): wd[j] = w[j] + k3[j]
        for j in range(m): k4[j] = h*fjtu[j](t+h,*wd)
        for j in range(m): w[j] = w[j]+(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6
        t = a + round(i*h,3)
        ti.append(t)
        xi.append(w)
    return(ti,xi)

# Modifed euler's method for a system
def mesys(a,b,N,alpham,fjtu):
    m = len(alpham)
    h = (b-a)/N
    t = a
    wd,error,w = [],[],[]
    for i in range(m):
        wd.append(0);error.append(0);w.append(alpham[i])
    ti=[a]
    xi=[w]
    for i in range(1,N+1):
        for j in range(m): wd[j] = w[j]+h*fjtu[j](t,*w)
        for j in range(m): w[j] = w[j]+h*fjtu[j](t,*wd)
        t = a + round(i*h,3)
        ti.append(t)
        xi.append(w)
    return(ti,xi)

# This finds the determinant via converting to triangular
# It also counts the operations and gives the permutation matrix
def ltdet(A):
    n = len(A)
    B = matcopy(A)
    m = matcopy(A)
    perm = matinit(n,n)
    p = -1
    x = []
    oc = 0
    det = 1
    for k in range(0,n):
        x.append(0)
    for i in range(0,n-1):
        p = -1
        for k in range(i,n):
            if(B[k][i]!=0 and p==-1):
                p = k
        if p == -1:
            print("No unique solution exists")
            return None
        if p != i:
            rowswap(B,p,i)
            rowswap(perm,p,i)
            oc = oc + 1
        for j in range(i+1,n):
            m[j][i] = B[j][i] / B[i][i]
            rowaddmult(B,i,j,-m[j][i])
            oc = oc + 1
    for i in range(0,n):
        det = det * B[i][i]
    det = det*(-1)**(oc+1)
    return(round(det,4),oc,perm)

# Gaussian elimination with backward substitution
def gebs(A,verbose):
    n = len(A)
    B = matcopy(A)
    m = matcopy(A)
    p = -1
    x = []
    numrowswaps = 0
    for k in range(0,n):
        x.append(0)
    for i in range(0,n-1):
        p = -1
        for k in range(i,n):
            if(B[k][i]!=0 and p==-1):
                p = k
        if p == -1:
            print("No unique solution exists")
            return None
        if p != i:
            numrowswaps = numrowswaps + 1
            rowswap(B,p,i)
        for j in range(i+1,n):
            m[j][i] = B[j][i] / B[i][i]
            rowaddmult(B,i,j,-m[j][i])
    if(B[n-1][n-1]==0):
        print("No unique solution exists")
        return None
    x[n-1] = B[n-1][n] / B[n-1][n-1]
    for i in range(n-2,-1,-1):
        summation = 0
        for k in range(i+1,n):
            summation = summation + B[i][k] * x[k]
        x[i] = (B[i][n] - summation) / B[i][i]
    if verbose: print("Number of row swaps: ", numrowswaps)
    return x

# Gaussian Elimination with partial pivoting
def gepp(A,verbose):
    summation = 0
    n = len(A)
    B = matcopy(A)
    C = matcopy(A)
    maximum = -1
    p = -1
    x = []
    numrowswaps = 0
    for k in range(0,n):
        x.append(0)
    for i in range(0,n-1):
        maximum = -1
        p = -1
        for k in range(i,n):
            if(abs(B[k][i])>maximum):
                maximum = abs(B[k][i])
                p = k
        if B[p][k] == 0:
            print("No unique solution exists")
            return None
        if i != p:
            numrowswaps = numrowswaps + 1
            rowswap(B,p,i)
        for j in range(i+1,n):
            C[j][i] = B[j][i] / B[i][i]
            rowaddmult(B,i,j,-C[j][i])
    if(B[n-1][n-1] == 0):
        print("No unique solution exists")
        return None
    x[n-1] = B[n-1][n] / B[n-1][n-1]
    for i in range(n-2,-1,-1):
        summation = 0
        for k in range(i+1,n):
            summation = summation + B[i][k] * x[k]
        x[i] = (B[i][n] - summation) / B[i][i]
    if verbose: print("Number of row swaps: ", numrowswaps)
    return x

# Gaussian elimination with scaled partial pivoting
def gespp(A,verbose):
    summation = 0
    n = len(A)
    B = matcopy(A)
    C = matcopy(A)
    maximum = -1
    p = -1
    x = []
    s1 = []
    numrowswaps = 0
    for k in range(0,n):
        s1.append(0)
        for l in range(0,n):
            if(abs(B[k][l])>s1[k]):
                s1[k] = abs(B[k][l])
        if(s1[k]==0):
            print("No unique solution exists")
            return None
        x.append(0)
    for i in range(0,n-1):
        maximum = -1
        p = -1
        for k in range(i,n):
            if(abs(B[k][i])/s1[k]>maximum):
                maximum = abs(B[k][i])/s1[k]
                p = k
        if B[p][k] == 0:
            print("No unique solution exists")
            return None
        if i != p:
            numrowswaps = numrowswaps + 1
            rowswap(B,p,i)
        for j in range(i+1,n):
            C[j][i] = B[j][i] / B[i][i]
            rowaddmult(B,i,j,-C[j][i])
    if(B[n-1][n-1] == 0):
        print("No unique solution exists")
        return None
    x[n-1] = B[n-1][n] / B[n-1][n-1]
    for i in range(n-2,-1,-1):
        summation = 0
        for k in range(i+1,n):
            summation = summation + B[i][k] * x[k]
        x[i] = (B[i][n] - summation) / B[i][i]
    if verbose: print("Number of row swaps: ", numrowswaps)
    return x

# Copys a matrix
def matcopy(A):
    B = [x[:] for x in A]
    return B

# Scales a matrix
def matscale(A,x):
    B = matcopy(A)
    for i in range(len(B)):
        for j in range(len(B[0])):
            B[i][j] = x*B[i][j]
    return B

# Row reduces into upper triangular
# Returns the row reduction and the upper triangular matrix
def rrut(A):
    n = len(A)
    B = matcopy(A)
    m = matcopy(A)
    p = -1
    x = []
    perm = []
    for k in range(n):
        x.append(0)
        perm.append([])
        for l in range(n):
            if k == l:
                perm[k].append(1)
            else:
                perm[k].append(0)
    for i in range(0,n-1):
        p = -1
        for k in range(i,n):
            if(B[k][i]!=0 and p==-1):
                p = k
        if p == -1:
            print("No unique solution exists")
            return None
        if p != i:
            rowswap(B,p,i)
            rowswap(perm,p,i)
        for j in range(i+1,n):
            m[j][i] = B[j][i] / B[i][i]
            rowaddmult(B,i,j,-m[j][i])
    return(B,perm)

# Solves with LU factorization
def LUsolve(A,b):
    n = len(A)
    U,P = rrut(A)
    L = matmult(matmult(P,A),inverse(U))
    y = []
    x = []
    for i in range(n):
        y.append(0)
        x.append(0)
    y[0] = b[0]/L[0][0]
    tot = 0
    for i in range(1,n):
        tot = 0
        for j in range(0,i):
            tot = tot + L[i][j] * y[j]
        y[i] = (1/L[i][i])*(b[i]- tot)
    x[n-1] = y[n-1]/U[n-1][n-1]
    for i in range(n-2,-1,-1):
        tot = 0
        for j in range(i+1,n):
            tot = tot + U[i][j] * x[j]
        x[i] = (1/U[i][i])*(y[i]-tot)
    return(x)

# Multiplies two matrices
def matmult(A,B):
    C = []
    n1 = len(A)
    m1 = len(A[0])
    n2 = len(B)
    m2 = len(B[0])
    element = 0
    if(m1!=n2):
        print("Incorrect dimensions")
        return None
    for _ in range(n1):
        C.append([])
    for i in range(n1):
        for j in range(m2):
            element = 0
            for k in range(m1):
                element = element + A[i][k]*B[k][j]
            C[i].append(element)
    return C

# Displays a matrix
def disa(A):
    if isinstance(A[0],list):
        for i in range(len(A)):
            print('[',end=' ')
            for j in range(len(A[0])):
                print(round(A[i][j],4),end=' ')
            print(']')
    else:
        print("[",end=' ')
        for i in range(len(A)):
            print(round(A[i],4),end=' ')
        print("]")
    return A

# Swaps rows of a matrix
def rowswap(A,e0,e1):
    rowtemp = A[e0]
    A[e0] = A[e1]
    A[e1] = rowtemp
    return A

# Performs a multiplied addition of another row operation
def rowaddmult(A,xsource,dest,x=1):
    for i in range(len(A[0])):
        A[dest][i] = A[dest][i] + x*A[xsource][i]
    return A

# Finds the determinant by finding the minor matrices
def det(A):
    n = len(A)
    deta = 0
    if(n==2):
        return(A[0][0]*A[1][1]-A[0][1]*A[1][0])
    for j in range(n):
        detm = det(constructminor(A,0,j))
        deta = deta + (-1)**(j)*A[0][j]*detm
    return deta

# Constructs the minor matrix
def constructminor(A,i,j):
    M = matcopy(A)
    for k in range(len(M)):
        del M[k][j]
    del M[i]
    return M

# Transposes a matrix
def transpose(A):
    B = matcopy(A)
    n = len(B)
    m = len(B[0])
    T = []
    for _ in range(m-1,-1,-1):
        T.append([])
    for j in range(m-1,-1,-1):
        for k in range(0,n):
            T[j].append(B[k][j])
    return T

# Finds the cofactor of a matrix
def cofactor(A):
    B = matcopy(A)
    n = len(B)
    m = len(B)
    for i in range(n):
        for j in range(m):
            B[i][j] = (-1)**(i+j)*det(constructminor(A,i,j))
    return B

# Finds the inverse of a matrix
def inverse(A):
    if det(A) != 0:
        return(matscale(transpose(cofactor(A)),(1/det(A))))
    else:
        print("No inverse found")
        return None

# Finds the LU decomposition of a matrix
def LU(A,lowdiag1s=True):
    n = len(A)
    u = matinit(n,n)
    l = matinit(n,n)
    tot = 0
    l[0][0] = 1 if lowdiag1s else A[0][0]
    u[0][0] = A[0][0] if lowdiag1s else 1
    if(A[0][0]==0):
        print("Factorization impossible")
        return
    for j in range(1,n):
        u[0][j] = A[0][j] / l[0][0]
        l[j][0] = A[j][0] / u[0][0]
    for i in range(1,n-1):
        tot = 0
        for k in range(i):
            tot = tot + l[i][k] * u[k][i]
        u[i][i] = A[i][i] - tot if lowdiag1s else 1
        l[i][i] = 1 if lowdiag1s else A[i][i] - tot
        if(l[i][i]*u[i][i]==0):
            print("Factorization impossibile")
            return
        for j in range(i+1,n):
            tot = 0
            for k in range(0,i):
                tot = tot + l[i][k] * u[k][j]
            u[i][j] = (1/l[i][i]) * (A[i][j] - tot)
            tot = 0
            for k in range(0,i):
                tot = tot + l[j][k] * u[k][i]
            l[j][i] = (1/u[i][i]) * (A[j][i] - tot)
    tot = 0
    for k in range(n-1):
        tot = tot + l[n-1][k] * u[k][n-1]
    l[n-1][n-1] = 1 if lowdiag1s else A[n-1][n-1] - tot
    u[n-1][n-1] = A[n-1][n-1] - tot if lowdiag1s else 1
    return l,u

# Finds the LDLt factorization
def LDLt(A):
    B = [x[:] for x in A]
    n = len(B)
    v = []
    d = []
    l = []
    for i in range(n):
        v.append(0)
        d.append(0)
        l.append([])
        for _ in B[0]:
            l[i].append(0)
    tot = 0
    for i in range(n):
        for j in range(i):
            v[j] = l[i][j]*d[j]
        tot = 0
        for k in range(i):
            tot = tot + l[i][k]*v[k]
        d[i] = B[i][i] - tot
        for j in range(i,n):
            tot = 0
            for k in range(i):
                tot = tot + l[j][k]*v[k]
            l[j][i] = (B[j][i] - tot)/d[i]
    return(l,dcompose(d),transpose(l))

# Finds the cholesky decomposition
# Or known as LLt
def cholesky(A):
    B = [x[:] for x in A]
    n = len(B)
    l = []
    tot = 0
    for i in range(n):
        l.append([])
        for _ in B[0]:
            l[i].append(0)
    l[0][0] = np.sqrt(B[0][0])
    for j in range(1,n):
        l[j][0] = B[j][0] / l[0][0]
    for i in range(1,n-1):
        tot = 0
        for k in range(i):
            tot = tot + (l[i][k])**2
        l[i][i] = (B[i][i] - tot)**(1/2)
        for j in range(i+1,n):
            tot = 0
            for k in range(i):
                tot = tot + l[j][k] * l[i][k]
            l[j][i] = (B[j][i] - tot) / l[i][i]
    tot = 0
    for k in range(n-1):
        tot = tot + (l[n-1][k])**2
    l[n-1][n-1] = (B[n-1][n-1] - tot)**(1/2)
    return(l,transpose(l))

# Creates m rows by n columns matrix
def matinit(m,n):
    mat = []
    for i in range(m):
        mat.append([])
        for _ in range(n):
            mat[i].append(0)
    return mat

# Creates a matrix with the list d as the diagonal
def dcompose(d):
    n = len(d)
    dmat = []
    for i in range(n):
        dmat.append([])
        for j in range(n):
            if i==j:
                dmat[i].append(d[i])
            else:
                dmat[i].append(0)
    return dmat