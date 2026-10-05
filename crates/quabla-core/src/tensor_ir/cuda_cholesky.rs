use super::super::CholeskyAdKind;

pub(super) fn scratch_lanes(kind: CholeskyAdKind) -> usize {
    let lanes = match kind {
        CholeskyAdKind::Vjp => 1,
        CholeskyAdKind::Mixed => 4,
        _ => 2,
    };
    lanes
        * match kind {
            CholeskyAdKind::Vjp => 2,
            CholeskyAdKind::VjpJvp => 3,
            _ => 1,
        }
}

pub(super) fn primal_source(function: &str, n: usize) -> String {
    format!(
        r#"extern "C" __global__ void {function}(const float* input, float* out, unsigned long long count) {{
    const unsigned long long offset = (unsigned long long)blockIdx.x * {n}ULL * {n}ULL;
    input += offset; out += offset; count = {n}ULL * {n}ULL;
    for (unsigned long long i = threadIdx.x; i < count; i += blockDim.x) out[i] = 0.0f;
    __syncthreads();
    for (unsigned long long column = 0; column < {n}ULL; ++column) {{
        if (threadIdx.x == 0U) {{
            float reduced = input[column * {n}ULL + column];
            for (unsigned long long k = 0; k < column; ++k) reduced = __fsub_rn(reduced, __fmul_rn(out[column * {n}ULL + k], out[column * {n}ULL + k]));
            out[column * {n}ULL + column] = sqrtf(reduced);
        }}
        __syncthreads();
        for (unsigned long long row = column + 1 + threadIdx.x; row < {n}ULL; row += blockDim.x) {{
            float reduced = input[row * {n}ULL + column];
            for (unsigned long long k = 0; k < column; ++k) reduced = __fsub_rn(reduced, __fmul_rn(out[row * {n}ULL + k], out[column * {n}ULL + k]));
            out[row * {n}ULL + column] = __fdiv_rn(reduced, out[column * {n}ULL + column]);
        }}
        __syncthreads();
    }}
}}
"#
    )
}

pub(super) fn ad_source(function: &str, n: usize, kind: CholeskyAdKind) -> String {
    let first = kind != CholeskyAdKind::Vjp;
    let mixed = kind == CholeskyAdKind::Mixed;
    let reverse = matches!(kind, CholeskyAdKind::Vjp | CholeskyAdKind::VjpJvp);
    let prefix = format!("{function}_helpers");
    let fields = if mixed {
        "float v,d,e,h;"
    } else if first {
        "float v,d;"
    } else {
        "float v;"
    };
    let lanes = if mixed {
        4
    } else if first {
        2
    } else {
        1
    };
    let mut source = format!("namespace {prefix} {{\nstruct Jet {{ {fields} }};\nstatic_assert(sizeof(Jet)=={}U, \"Jet layout\");\n",lanes*4);
    source.push_str("__device__ float addf(float a,float b){return __fadd_rn(a,b);}\n__device__ float mulf(float a,float b){return __fmul_rn(a,b);}\n__device__ float subf(float a,float b){return __fsub_rn(a,b);}\n__device__ float divf(float a,float b){return __fdiv_rn(a,b);}\n");
    source.push_str("__device__ Jet add(Jet a,Jet b){Jet c;c.v=addf(a.v,b.v);");
    if first {
        source.push_str("c.d=addf(a.d,b.d);");
    }
    if mixed {
        source.push_str("c.e=addf(a.e,b.e);c.h=addf(a.h,b.h);");
    }
    source.push_str("return c;}\n__device__ Jet neg(Jet a){a.v=-a.v;");
    if first {
        source.push_str("a.d=-a.d;");
    }
    if mixed {
        source.push_str("a.e=-a.e;a.h=-a.h;");
    }
    source.push_str("return a;}\n__device__ Jet sub(Jet a,Jet b){return add(a,neg(b));}\n__device__ Jet mul(Jet a,Jet b){Jet c;c.v=mulf(a.v,b.v);");
    if first {
        source.push_str("c.d=addf(mulf(a.d,b.v),mulf(a.v,b.d));");
    }
    if mixed {
        source.push_str("c.e=addf(mulf(a.e,b.v),mulf(a.v,b.e));c.h=addf(addf(mulf(a.h,b.v),mulf(a.d,b.e)),addf(mulf(a.e,b.d),mulf(a.v,b.h)));");
    }
    source.push_str("return c;}\n__device__ Jet divide(Jet a,Jet b){Jet c;c.v=divf(a.v,b.v);float denominator=mulf(b.v,b.v);\n");
    if first {
        source.push_str(
            "float numerator=subf(mulf(a.d,b.v),mulf(a.v,b.d));c.d=divf(numerator,denominator);",
        );
    }
    if mixed {
        source.push_str("c.e=divf(subf(mulf(a.e,b.v),mulf(a.v,b.e)),denominator);float numerator_second=subf(addf(mulf(a.h,b.v),mulf(a.d,b.e)),addf(mulf(a.e,b.d),mulf(a.v,b.h)));float denominator_second=addf(mulf(b.e,b.v),mulf(b.v,b.e));c.h=divf(subf(mulf(numerator_second,denominator),mulf(numerator,denominator_second)),mulf(denominator,denominator));");
    }
    source.push_str("return c;}\n__device__ float derivative(float x,int order){if(x==0.0f)return 0.0f;if(x<0.0f)return __int_as_float(0x7fc00000);float coefficient=1.0f;for(int i=0;i<order;++i)coefficient=mulf(coefficient,0.5f-(float)i);return mulf(coefficient,powf(x,0.5f-(float)order));}\n__device__ Jet root(Jet a,int order){Jet c;c.v=order==0?sqrtf(a.v):derivative(a.v,order);");
    if first {
        source.push_str("c.d=mulf(a.d,derivative(a.v,order+1));");
    }
    if mixed {
        source.push_str("c.e=mulf(a.e,derivative(a.v,order+1));c.h=addf(mulf(a.h,derivative(a.v,order+1)),mulf(a.d,mulf(a.e,derivative(a.v,order+2))));");
    }
    source.push_str("return c;}\n}\n");
    let parameters = if matches!(kind, CholeskyAdKind::Jvp | CholeskyAdKind::Vjp) {
        "const float* input,const float* other"
    } else {
        "const float* input,const float* other,const float* third,const float* fourth"
    };
    source.push_str(&format!("extern \"C\" __global__ void {function}({parameters},float* out,float* scratch){{\nusing namespace {prefix};\nif(threadIdx.x!=0U)return;\nconst unsigned long long n={n}ULL,count=n*n;\nJet* values=(Jet*)scratch;\n"));
    // Every block owns one independent leading-batch matrix and its Jet tape.
    source.push_str(&format!("unsigned long long batch_offset=(unsigned long long)blockIdx.x*count;input+=batch_offset;other+=batch_offset;out+=batch_offset;values+=(unsigned long long)blockIdx.x*count*{};\n", match kind { CholeskyAdKind::Vjp => 2, CholeskyAdKind::VjpJvp => 3, _ => 1 }));
    if !matches!(kind, CholeskyAdKind::Jvp | CholeskyAdKind::Vjp) {
        source.push_str("third+=batch_offset;fourth+=batch_offset;\n");
    }
    if reverse {
        source.push_str("Jet* residuals=values+count;\n");
        // Descending reverse traversal never revisits a consumed cotangent.
        // Reuse the F32 VJP output as that cotangent's final gradient storage.
        source.push_str(if kind == CholeskyAdKind::Vjp {
            "Jet* cotangents=(Jet*)out;\n"
        } else {
            "Jet* cotangents=residuals+count;\n"
        });
    }
    source.push_str("for(unsigned long long i=0;i<count;++i){values[i]=Jet{};out[i]=0.0f;");
    if reverse {
        let upstream = if kind == CholeskyAdKind::Vjp {
            "(i%n<=i/n?other[i]:0.0f)"
        } else {
            "third[i]"
        };
        source.push_str(&format!(
            "cotangents[i]=Jet{{}};cotangents[i].v={upstream};"
        ));
        if first {
            source.push_str("cotangents[i].d=fourth[i];");
        }
    }
    source.push_str("}\nfor(unsigned long long row=0;row<n;++row){for(unsigned long long col=0;col<=row;++col){unsigned long long i=row*n+col;Jet reduced{};reduced.v=input[i];");
    if first {
        source.push_str("reduced.d=other[i];");
    }
    if mixed {
        source.push_str("reduced.e=third[i];reduced.h=fourth[i];");
    }
    source.push_str("for(unsigned long long k=0;k<col;++k)reduced=sub(reduced,mul(values[row*n+k],values[col*n+k]));");
    if reverse {
        source.push_str("residuals[i]=reduced;");
    }
    source.push_str("values[i]=row==col?root(reduced,0):divide(reduced,values[col*n+col]);");
    if !reverse {
        source.push_str(if mixed {
            "out[i]=values[i].h;"
        } else {
            "out[i]=values[i].d;"
        });
    }
    source.push_str("}}\n");
    if reverse {
        source.push_str("for(unsigned long long r=n;r>0;--r){unsigned long long row=r-1;for(unsigned long long c=row+1;c>0;--c){unsigned long long col=c-1,i=row*n+col;Jet upstream=cotangents[i],reduced;if(row==col){reduced=mul(upstream,root(residuals[i],1));}else{unsigned long long diagonal=col*n+col;Jet contribution=neg(divide(mul(upstream,residuals[i]),mul(values[diagonal],values[diagonal])));cotangents[diagonal]=add(cotangents[diagonal],contribution);reduced=divide(upstream,values[diagonal]);}");
        source.push_str(if first {
            "out[i]=reduced.d;"
        } else {
            "out[i]=reduced.v;"
        });
        source.push_str("for(unsigned long long k=col;k>0;--k){unsigned long long inner=k-1,left=row*n+inner,right=col*n+inner;Jet product=neg(reduced);cotangents[left]=add(cotangents[left],mul(product,values[right]));cotangents[right]=add(cotangents[right],mul(product,values[left]));}}}\n");
    }
    source.push_str("}\n");
    source
}
