import { useState, useRef } from "react";

const TMDB     = "https://image.tmdb.org/t/p/w300";
const TMDB_BIG = "https://image.tmdb.org/t/p/w780";

const ALL_MOVIES = [
  { id:1,  title:"Inception",         year:2010, rating:8.3, runtime:148, budget:"$160M", revenue:"$836M",
    genres:["Science Fiction","Thriller"], poster:"/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg", backdrop:"/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
    director:"Christopher Nolan", language:"English", status:"Released",
    cast:["Leonardo DiCaprio","Joseph Gordon-Levitt","Elliot Page","Tom Hardy","Ken Watanabe"],
    overview:"A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O. — but his tragic past may doom the project and his team to disaster.",
    tagline:"Your mind is the scene of the crime.",
    keywords:["dream","subconscious","heist","psychological","sci-fi"] },
  { id:2,  title:"Interstellar",       year:2014, rating:8.1, runtime:169, budget:"$165M", revenue:"$701M",
    genres:["Science Fiction","Drama"], poster:"/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg", backdrop:"/pbrkL804wmCVN334e4GElPdJ1LB.jpg",
    director:"Christopher Nolan", language:"English", status:"Released",
    cast:["Matthew McConaughey","Anne Hathaway","Jessica Chastain","Matt Damon","Michael Caine"],
    overview:"The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.",
    tagline:"Mankind was born on Earth. It was never meant to die here.",
    keywords:["space","wormhole","time","black hole","NASA"] },
  { id:3,  title:"The Dark Knight",    year:2008, rating:9.0, runtime:152, budget:"$185M", revenue:"$1B",
    genres:["Action","Crime"], poster:"/qJ2tW6WMUDux911r6m7haRef0WH.jpg", backdrop:"/hqkIcbrOHL86UncnHIsHVcVmzue.jpg",
    director:"Christopher Nolan", language:"English", status:"Released",
    cast:["Christian Bale","Heath Ledger","Aaron Eckhart","Maggie Gyllenhaal","Gary Oldman"],
    overview:"When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
    tagline:"Why so serious?",
    keywords:["batman","joker","gotham","superhero","vigilante"] },
  { id:4,  title:"Parasite",           year:2019, rating:8.6, runtime:132, budget:"$11M", revenue:"$263M",
    genres:["Drama","Thriller"], poster:"/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg", backdrop:"/TU9NIjwzjoKPwQHoHshkFcQUCG.jpg",
    director:"Bong Joon-ho", language:"Korean", status:"Released",
    cast:["Song Kang-ho","Lee Sun-kyun","Cho Yeo-jeong","Choi Woo-shik","Park So-dam"],
    overview:"All unemployed, Ki-taek's family takes a special interest in the wealthy Park family. As they scheme and infiltrate the Parks' lives, a grim reality unfolds revealing the complex layers of class and privilege.",
    tagline:"Act like you own the place.",
    keywords:["class divide","family","dark comedy","social satire","poverty"] },
  { id:5,  title:"Arrival",            year:2016, rating:7.9, runtime:116, budget:"$47M", revenue:"$203M",
    genres:["Science Fiction","Drama"], poster:"/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg", backdrop:"/k3yABPHEeJ3WZAHB7fYhYASXKA7.jpg",
    director:"Denis Villeneuve", language:"English", status:"Released",
    cast:["Amy Adams","Jeremy Renner","Forest Whitaker","Michael Stuhlbarg","Mark O'Brien"],
    overview:"When mysterious spacecraft touch down across the globe, an elite team led by expert linguist Louise Banks is brought together to investigate. As nations teeter on the verge of global war, Banks must risk everything.",
    tagline:"Why are they here?",
    keywords:["aliens","language","time","first contact","linguistics"] },
  { id:6,  title:"Blade Runner 2049",  year:2017, rating:8.0, runtime:164, budget:"$150M", revenue:"$260M",
    genres:["Science Fiction","Thriller"], poster:"/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg", backdrop:"/ilRyazdMfOUdzulQgOrm4bNGQJf.jpg",
    director:"Denis Villeneuve", language:"English", status:"Released",
    cast:["Ryan Gosling","Harrison Ford","Ana de Armas","Sylvia Hoeks","Robin Wright"],
    overview:"Thirty years after the events of the first film, a new blade runner unearths a long-buried secret that has the potential to plunge what's left of society into chaos.",
    tagline:"The key to the future is finally unearthed.",
    keywords:["replicants","dystopia","cyberpunk","identity","future"] },
  { id:7,  title:"Whiplash",           year:2014, rating:8.5, runtime:107, budget:"$3.3M", revenue:"$49M",
    genres:["Drama","Music"], poster:"/7fn624j5lj3xTme2SgiLCeuedmO.jpg", backdrop:"/6bbZ6XyvgfjhQwbplnUh1LSj1ky.jpg",
    director:"Damien Chazelle", language:"English", status:"Released",
    cast:["Miles Teller","J.K. Simmons","Melissa Benoist","Paul Reiser","Austin Stowell"],
    overview:"A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's full potential.",
    tagline:"The road to greatness can take you to the edge.",
    keywords:["music","jazz","ambition","obsession","conservatory"] },
  { id:8,  title:"The Prestige",       year:2006, rating:8.5, runtime:130, budget:"$40M", revenue:"$110M",
    genres:["Drama","Mystery"], poster:"/5MXyQfz8xUP3dIFPTubhTsbFY6V.jpg", backdrop:"/rfK5LHEAnJNJGVWMCnH9WCCG8bK.jpg",
    director:"Christopher Nolan", language:"English", status:"Released",
    cast:["Hugh Jackman","Christian Bale","Michael Caine","Scarlett Johansson","David Bowie"],
    overview:"Two stage magicians engage in competitive one-upmanship in an attempt to create the ultimate stage illusion. Obsession, betrayal, and sacrifice define their dangerous rivalry.",
    tagline:"Are you watching closely?",
    keywords:["magic","obsession","rivalry","illusion","victorian"] },
  { id:9,  title:"Ex Machina",         year:2014, rating:7.7, runtime:108, budget:"$15M", revenue:"$37M",
    genres:["Science Fiction","Drama"], poster:"/btmgCKFhQJEOPOtIUZBCnEKbHDm.jpg", backdrop:"/9GvhICFMiRQA72iyc3KgZnMeRF7.jpg",
    director:"Alex Garland", language:"English", status:"Released",
    cast:["Domhnall Gleeson","Oscar Isaac","Alicia Vikander","Sonoya Mizuno"],
    overview:"A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid A.I. named Ava.",
    tagline:"What happens to me if I fail your test?",
    keywords:["AI","Turing test","consciousness","robot","isolation"] },
  { id:10, title:"The Martian",        year:2015, rating:7.6, runtime:144, budget:"$108M", revenue:"$630M",
    genres:["Science Fiction","Drama"], poster:"/5aGhaIHYuQbqlHWvWYqMCnj40y2.jpg", backdrop:"/sy3e2e4JwdAtd2oZGA2uIxTSZw9.jpg",
    director:"Ridley Scott", language:"English", status:"Released",
    cast:["Matt Damon","Jessica Chastain","Kristen Wiig","Jeff Daniels","Michael Peña"],
    overview:"During a manned mission to Mars, Astronaut Mark Watney is presumed dead and left behind. Stranded alone on the hostile planet, he must draw upon his ingenuity and spirit to survive.",
    tagline:"Help is only 140 million miles away.",
    keywords:["mars","survival","astronaut","NASA","botany"] },
  { id:11, title:"Gravity",            year:2013, rating:7.0, runtime:91,  budget:"$100M", revenue:"$723M",
    genres:["Science Fiction","Thriller"], poster:"/jVBKEjRcGDRGJt4L9NtaRJaHSZQ.jpg", backdrop:"/3qHGSgNrTmXh8HY3f3WMYTM34l0.jpg",
    director:"Alfonso Cuarón", language:"English", status:"Released",
    cast:["Sandra Bullock","George Clooney","Ed Harris","Orto Ignatiussen"],
    overview:"Dr. Ryan Stone is a brilliant medical engineer on her first shuttle mission. On a routine spacewalk disaster strikes and she is adrift in space with no way home.",
    tagline:"Don't let go.",
    keywords:["space","survival","astronaut","debris","orbit"] },
  { id:12, title:"Joker",              year:2019, rating:8.4, runtime:122, budget:"$55M", revenue:"$1.07B",
    genres:["Crime","Drama"], poster:"/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg", backdrop:"/n6bUvigpRFqSwmPp1m2YADdbRBc.jpg",
    director:"Todd Phillips", language:"English", status:"Released",
    cast:["Joaquin Phoenix","Robert De Niro","Zazie Beetz","Frances Conroy","Brett Cullen"],
    overview:"In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime.",
    tagline:"Put on a happy face.",
    keywords:["gotham","mental illness","clown","revolution","origin story"] },
  { id:13, title:"Oppenheimer",        year:2023, rating:8.3, runtime:180, budget:"$100M", revenue:"$952M",
    genres:["Drama","History"], poster:"/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg", backdrop:"/fm6KqXpk3M2HVveHwCrBSSBaO0V.jpg",
    director:"Christopher Nolan", language:"English", status:"Released",
    cast:["Cillian Murphy","Emily Blunt","Matt Damon","Robert Downey Jr.","Florence Pugh"],
    overview:"The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II.",
    tagline:"The world forever changes.",
    keywords:["atomic bomb","Manhattan Project","physics","WWII","biography"] },
  { id:14, title:"Dune",               year:2021, rating:7.9, runtime:155, budget:"$165M", revenue:"$401M",
    genres:["Science Fiction","Adventure"], poster:"/d5NXSklpcvwN3Y1RQ58yaoxoWeI.jpg", backdrop:"/jYEW5xZkZk2WTrdbMGAPFuBqbDc.jpg",
    director:"Denis Villeneuve", language:"English", status:"Released",
    cast:["Timothée Chalamet","Rebecca Ferguson","Oscar Isaac","Josh Brolin","Stellan Skarsgård"],
    overview:"Feature adaptation of Frank Herbert's science fiction novel about the son of a noble family entrusted with the protection of the most valuable asset in the galaxy on a desert planet.",
    tagline:"Beyond fear, destiny awaits.",
    keywords:["desert","spice","prophecy","noble houses","sandworm"] },
  { id:15, title:"The Shawshank Redemption", year:1994, rating:9.3, runtime:142, budget:"$25M", revenue:"$58M",
    genres:["Drama","Crime"], poster:"/lyQBXzOQSuE59IsHyhrp0qIiPAz.jpg", backdrop:"/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg",
    director:"Frank Darabont", language:"English", status:"Released",
    cast:["Tim Robbins","Morgan Freeman","Bob Gunton","William Sadler","Clancy Brown"],
    overview:"Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. A timeless story of hope, friendship and the human spirit.",
    tagline:"Fear can hold you prisoner. Hope can set you free.",
    keywords:["prison","hope","friendship","redemption","escape"] },
  { id:16, title:"Fight Club",         year:1999, rating:8.8, runtime:139, budget:"$63M", revenue:"$101M",
    genres:["Drama","Thriller"], poster:"/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg", backdrop:"/87hTDiay2N2qWyX4Ds7wmrm3ire.jpg",
    director:"David Fincher", language:"English", status:"Released",
    cast:["Brad Pitt","Edward Norton","Helena Bonham Carter","Meat Loaf","Jared Leto"],
    overview:"An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much more. A raw exploration of masculinity and consumerism.",
    tagline:"Mischief. Mayhem. Soap.",
    keywords:["masculinity","consumerism","identity","anarchy","underground"] },
];

const GENRES    = ["Action","Animation","Comedy","Crime","Drama","History","Horror","Mystery","Romance","Science Fiction","Thriller"];
const INTERESTS = ["Sci-Fi","Drama","Thriller","Comedy","Action","Horror","Romance","Mystery","Animation","Crime","History","Documentary"];
const GENRE_STATS = [
  {label:"Drama",pct:95,count:2297},{label:"Comedy",pct:66,count:1603},{label:"Thriller",pct:55,count:1323},
  {label:"Action",pct:48,count:1154},{label:"Romance",pct:36,count:860},{label:"Adventure",pct:33,count:793},
  {label:"Crime",pct:31,count:760},{label:"Sci-Fi",pct:27,count:658},
];

function ts() { return new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}); }
function generateCode() { return String(Math.floor(100000+Math.random()*900000)); }
function estimateRevenue(b,p,rt,r,v) { return Math.max(0,b*2.1e6+p*800000+(r-5)*12e6+v*800); }

const GoogleIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24">
    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
  </svg>
);

const Poster = ({src, alt, style={}}) => {
  const [err,setErr] = useState(false);
  return <img src={err?`https://placehold.co/300x450/1a1a1a/555?text=${encodeURIComponent(alt||'?')}`:TMDB+src}
    alt={alt} onError={()=>setErr(true)} style={{display:"block",objectFit:"cover",...style}}/>;
};

const BackdropImg = ({src, alt}) => {
  const [err,setErr] = useState(false);
  return <img
    src={err?`https://placehold.co/780x440/111/333?text=No+Backdrop`:TMDB_BIG+src}
    alt={alt} onError={()=>setErr(true)}
    style={{width:"100%",height:"100%",objectFit:"cover",display:"block"}}/>;
};

/* ═══════════════════════════════════════════════
   MOVIE DETAIL MODAL
═══════════════════════════════════════════════ */
function MovieModal({ movie, onClose, watched, onMarkWatched, onOpenSimilar }) {
  const isWatched = watched.includes(movie.id);
  const ratingCol = movie.rating>=8.5?"#2ecc71":movie.rating>=7.5?"#e0a84b":"#e07b5b";
  const similar   = ALL_MOVIES.filter(m=>m.id!==movie.id&&m.genres.some(g=>movie.genres.includes(g))).sort((a,b)=>b.rating-a.rating).slice(0,4);

  return (
    <div onClick={onClose} style={{
      position:"fixed",inset:0,background:"rgba(0,0,0,0.9)",zIndex:9999,
      display:"flex",alignItems:"center",justifyContent:"center",padding:20,
      backdropFilter:"blur(8px)",animation:"fadeInBg .2s ease"
    }}>
      <div onClick={e=>e.stopPropagation()} style={{
        background:"#141414",borderRadius:18,width:"100%",maxWidth:880,
        maxHeight:"90vh",overflowY:"auto",border:"1px solid #2a2a2a",
        animation:"modalUp .3s ease",position:"relative",
        boxShadow:"0 40px 100px rgba(0,0,0,0.85)"
      }}>
        {/* ── BACKDROP ── */}
        <div style={{position:"relative",height:320,borderRadius:"18px 18px 0 0",overflow:"hidden",flexShrink:0}}>
          <BackdropImg src={movie.backdrop} alt={movie.title}/>
          <div style={{position:"absolute",inset:0,background:"linear-gradient(to bottom,rgba(0,0,0,.05) 0%,rgba(20,20,20,.65) 55%,#141414 100%)"}}/>

          {/* Close */}
          <button onClick={onClose} style={{
            position:"absolute",top:14,right:14,background:"rgba(0,0,0,.65)",
            border:"1px solid #333",borderRadius:"50%",width:34,height:34,
            color:"#ddd",fontSize:16,cursor:"pointer",display:"flex",
            alignItems:"center",justifyContent:"center",transition:"all .15s",zIndex:10
          }} onMouseEnter={e=>{e.target.style.background="#333"}} onMouseLeave={e=>{e.target.style.background="rgba(0,0,0,.65)"}}>✕</button>

          {/* Poster + title */}
          <div style={{position:"absolute",bottom:0,left:0,right:0,padding:"0 26px 22px",display:"flex",gap:20,alignItems:"flex-end"}}>
            <div style={{flexShrink:0,borderRadius:9,overflow:"hidden",border:"2px solid rgba(224,168,75,.4)",boxShadow:"0 8px 28px rgba(0,0,0,.7)"}}>
              <Poster src={movie.poster} alt={movie.title} style={{width:100,height:150}}/>
            </div>
            <div style={{flex:1,paddingBottom:2}}>
              {movie.tagline&&<div style={{fontSize:11,color:"#e0a84b",fontStyle:"italic",marginBottom:4,opacity:.9}}>{movie.tagline}</div>}
              <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:34,letterSpacing:1,color:"#f0ece4",lineHeight:1.05}}>{movie.title}</div>
              <div style={{fontSize:12,color:"#888",marginTop:5,display:"flex",gap:10,flexWrap:"wrap",alignItems:"center"}}>
                <span>{movie.year}</span><span style={{color:"#333"}}>·</span>
                <span>{movie.runtime} min</span><span style={{color:"#333"}}>·</span>
                <span>{movie.language}</span><span style={{color:"#333"}}>·</span>
                <span style={{background:"#1e1e1e",border:"1px solid #2a2a2a",borderRadius:4,padding:"1px 7px",fontSize:10}}>{movie.status}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── BODY ── */}
        <div style={{padding:"22px 26px 30px"}}>

          {/* Rating + genres + CTA */}
          <div style={{display:"flex",alignItems:"center",gap:10,flexWrap:"wrap",marginBottom:18}}>
            <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:30,color:ratingCol,letterSpacing:1}}>★ {movie.rating}</div>
            <div style={{fontSize:11,color:"#555",marginRight:6}}>/10</div>
            {movie.genres.map(g=>(
              <span key={g} style={{padding:"4px 11px",background:"#1e1e1e",border:"1px solid #2a2a2a",borderRadius:20,fontSize:11,color:"#d0ccc5",fontWeight:600}}>{g}</span>
            ))}
            <button onClick={()=>onMarkWatched(movie)} style={{
              marginLeft:"auto",background:isWatched?"#0f1f0f":"#e0a84b",
              color:isWatched?"#5a9e5a":"#0c0c0c",
              border:isWatched?"1px solid #1a3a1a":"none",
              borderRadius:8,padding:"9px 18px",fontFamily:"'DM Sans',sans-serif",
              fontWeight:700,fontSize:12,cursor:"pointer",transition:"all .15s",flexShrink:0,
            }}>{isWatched?"✓ Already Watched":"+ Mark as Watched"}</button>
          </div>

          {/* Overview */}
          <p style={{fontSize:13,color:"#999",lineHeight:1.8,marginBottom:22,maxWidth:680}}>{movie.overview}</p>

          {/* Stats */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:22}}>
            {[{l:"Director",v:movie.director},{l:"Budget",v:movie.budget},{l:"Revenue",v:movie.revenue},{l:"Runtime",v:`${movie.runtime} min`}].map(s=>(
              <div key={s.l} style={{background:"#1a1a1a",border:"1px solid #222",borderRadius:9,padding:"13px 14px"}}>
                <div style={{fontSize:9,color:"#555",textTransform:"uppercase",letterSpacing:.8,marginBottom:5}}>{s.l}</div>
                <div style={{fontSize:13,fontWeight:700,color:"#f0ece4"}}>{s.v}</div>
              </div>
            ))}
          </div>

          {/* Cast */}
          <div style={{marginBottom:20}}>
            <div style={{fontSize:10,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:10,fontWeight:700}}>Cast</div>
            <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
              {movie.cast.map(name=>(
                <div key={name} style={{display:"flex",alignItems:"center",gap:8,background:"#1a1a1a",border:"1px solid #222",borderRadius:8,padding:"7px 11px"}}>
                  <div style={{width:26,height:26,borderRadius:"50%",background:"#2a2a2a",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,fontWeight:800,color:"#e0a84b",flexShrink:0}}>
                    {name.split(" ").map(w=>w[0]).join("").slice(0,2)}
                  </div>
                  <span style={{fontSize:11,color:"#d0ccc5",fontWeight:500,whiteSpace:"nowrap"}}>{name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Keywords */}
          <div style={{marginBottom:24}}>
            <div style={{fontSize:10,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:8,fontWeight:700}}>Keywords</div>
            <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
              {movie.keywords.map(k=>(
                <span key={k} style={{padding:"4px 10px",background:"#111",border:"1px solid #1e1e1e",borderRadius:20,fontSize:11,color:"#555"}}>{k}</span>
              ))}
            </div>
          </div>

          {/* Similar movies */}
          {similar.length>0&&(
            <div>
              <div style={{fontSize:10,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:12,fontWeight:700}}>You Might Also Like</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
                {similar.map(m=>(
                  <div key={m.id} onClick={()=>onOpenSimilar(m)}
                    style={{background:"#1a1a1a",border:"1px solid #1e1e1e",borderRadius:9,overflow:"hidden",cursor:"pointer",transition:"border-color .2s"}}
                    onMouseEnter={e=>e.currentTarget.style.borderColor="#e0a84b"}
                    onMouseLeave={e=>e.currentTarget.style.borderColor="#1e1e1e"}>
                    <Poster src={m.poster} alt={m.title} style={{width:"100%",aspectRatio:"2/3"}}/>
                    <div style={{padding:"8px 10px"}}>
                      <div style={{fontSize:11,fontWeight:700,color:"#f0ece4",marginBottom:2,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{m.title}</div>
                      <div style={{fontSize:10,color:"#555"}}>⭐ {m.rating} · {m.year}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   MOVIE CARD
═══════════════════════════════════════════════ */
function MovieCard({ movie, onOpen, watched, onMarkWatched, delay=0 }) {
  const isWatched = watched.includes(movie.id);
  const [hover, setHover] = useState(false);
  return (
    <div
      onClick={()=>onOpen(movie)}
      onMouseEnter={()=>setHover(true)}
      onMouseLeave={()=>setHover(false)}
      style={{
        background:"#141414",border:`1px solid ${hover?"#e0a84b":"#1e1e1e"}`,borderRadius:10,
        overflow:"hidden",cursor:"pointer",position:"relative",
        transform:hover?"translateY(-5px)":"translateY(0)",
        boxShadow:hover?"0 12px 30px rgba(224,168,75,.15)":"none",
        transition:"transform .2s,border-color .2s,box-shadow .2s",
        animationDelay:`${delay}s`,animation:"fadeSlideUp .3s ease forwards",opacity:0,
      }}>
      {isWatched&&<div style={{position:"absolute",top:7,right:7,zIndex:2,background:"rgba(10,10,10,.9)",border:"1px solid #2a3a2a",borderRadius:10,padding:"2px 7px",fontSize:9,color:"#5a9e5a",fontWeight:700}}>✓ Watched</div>}
      <div style={{position:"absolute",top:7,left:7,zIndex:2,background:"#e0a84b",color:"#0c0c0c",fontSize:9,fontWeight:800,padding:"2px 6px",borderRadius:4,letterSpacing:.5}}>HD</div>
      {hover&&<div style={{position:"absolute",inset:0,zIndex:1,background:"linear-gradient(to top,rgba(20,20,20,.95) 0%,rgba(0,0,0,.2) 60%)",display:"flex",alignItems:"flex-end",padding:"10px",pointerEvents:"none"}}>
        <div style={{fontSize:11,color:"#e0a84b",fontWeight:600}}>Click for details →</div>
      </div>}
      <Poster src={movie.poster} alt={movie.title} style={{width:"100%",aspectRatio:"2/3"}}/>
      <div style={{padding:"10px 11px 12px"}}>
        <div style={{fontSize:12,fontWeight:700,color:"#f0ece4",marginBottom:3,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{movie.title}</div>
        <div style={{fontSize:10,color:"#555",marginBottom:7}}>{movie.year} · ⭐ {movie.rating}</div>
        <button onClick={e=>{e.stopPropagation();onMarkWatched(movie);}} style={{
          background:isWatched?"#0f1f0f":"#1a1600",color:isWatched?"#5a9e5a":"#e0a84b",
          border:isWatched?"1px solid #1a3a1a":"1px solid #2e2200",
          borderRadius:6,padding:"4px 10px",fontSize:10,fontWeight:700,cursor:"pointer",
          fontFamily:"'DM Sans',sans-serif",width:"100%",transition:"all .15s"
        }}>{isWatched?"✓ Watched":"+ Mark Watched"}</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   CSS
═══════════════════════════════════════════════ */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:#0c0c0c;color:#f0ece4;min-height:100vh}
@keyframes fadeInBg{from{opacity:0}to{opacity:1}}
@keyframes modalUp{from{opacity:0;transform:translateY(24px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes fi{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes su{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
.fi{animation:fi .3s ease forwards}
.su{animation:su .4s ease forwards}
.app{display:flex;min-height:100vh}
.sidebar{width:224px;min-width:224px;background:#111;border-right:1px solid #1c1c1c;padding:22px 14px;display:flex;flex-direction:column}
.s-logo{font-family:'Bebas Neue',sans-serif;font-size:21px;color:#e0a84b;letter-spacing:2px;margin-bottom:3px}
.s-sub{font-size:10px;color:#444;margin-bottom:18px;line-height:1.6}
.s-label{font-size:10px;color:#3a3a3a;text-transform:uppercase;letter-spacing:1px;margin:10px 0 5px}
.nav{padding:9px 11px;border-radius:7px;font-size:13px;color:#777;cursor:pointer;display:flex;align-items:center;gap:9px;transition:all .15s;margin-bottom:1px}
.nav:hover{background:#1a1a1a;color:#ccc}
.nav.active{background:#1a1600;color:#e0a84b;font-weight:600}
.s-hr{border:none;border-top:1px solid #1a1a1a;margin:12px 0}
.user-chip{display:flex;align-items:center;gap:10px;padding:10px 11px;background:#1a1a1a;border-radius:9px;margin-top:auto;border:1px solid #1e1e1e}
.avatar{width:32px;height:32px;border-radius:50%;background:#e0a84b;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#0c0c0c;flex-shrink:0}
.u-name{font-size:13px;color:#d0ccc5;font-weight:600;line-height:1.2}
.u-email{font-size:10px;color:#444}
.signout{font-size:11px;color:#444;cursor:pointer;text-align:center;margin-top:8px;transition:color .15s}
.signout:hover{color:#e07b5b}
.main{flex:1;padding:28px 32px;overflow-y:auto;max-height:100vh}
.pg-title{font-family:'Bebas Neue',sans-serif;font-size:30px;color:#f0ece4;letter-spacing:1px;margin-bottom:4px}
.pg-sub{font-size:13px;color:#555;margin-bottom:22px}
.metrics{display:grid;gap:10px;margin-bottom:22px}
.mc{background:#141414;border:1px solid #1e1e1e;border-radius:11px;padding:16px;text-align:center}
.mv{font-family:'Bebas Neue',sans-serif;font-size:28px;color:#e0a84b;letter-spacing:1px;line-height:1}
.ml{font-size:11px;color:#555;margin-top:4px}
.hero{background:linear-gradient(135deg,#1a1200,#141414);border:1px solid #2e2200;border-radius:14px;padding:20px 24px;margin-bottom:22px;display:flex;align-items:center;gap:16px}
.hero h3{font-size:15px;font-weight:700;color:#e0a84b;margin-bottom:4px}
.hero p{font-size:12px;color:#777;line-height:1.5}
.sbar{display:flex;gap:10px;margin-bottom:14px;align-items:flex-end}
.ig{display:flex;flex-direction:column;gap:5px;flex:1}
.ilabel{font-size:11px;color:#777}
.input{background:#141414;border:1px solid #242424;border-radius:8px;padding:10px 13px;color:#f0ece4;font-family:'DM Sans',sans-serif;font-size:13px;outline:none;transition:border-color .2s;width:100%}
.input:focus{border-color:#e0a84b}
.input::placeholder{color:#333}
.btn{background:#e0a84b;color:#0c0c0c;border:none;border-radius:8px;padding:10px 20px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:12px;cursor:pointer;transition:background .15s;white-space:nowrap;height:38px}
.btn:hover{background:#f0b95c}
.btn-ol{background:transparent;color:#e0a84b;border:1px solid #e0a84b;border-radius:7px;padding:7px 13px;font-family:'DM Sans',sans-serif;font-size:11px;font-weight:700;cursor:pointer;transition:all .15s}
.btn-ol:hover{background:#1a1600}
.gpills{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:16px}
.gpill{padding:4px 11px;border-radius:20px;font-size:11px;border:1px solid #242424;background:#141414;color:#666;cursor:pointer;transition:all .15s}
.gpill:hover{border-color:#e0a84b;color:#e0a84b}
.gpill.sel{background:#1a1600;border-color:#e0a84b;color:#e0a84b}
.mgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.tabs{display:flex;border-bottom:1px solid #1e1e1e;margin-bottom:18px}
.tab{padding:8px 14px;font-size:12px;color:#555;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;font-weight:500}
.tab:hover{color:#ccc}
.tab.active{color:#e0a84b;border-bottom-color:#e0a84b}
.chart{background:#141414;border:1px solid #1e1e1e;border-radius:11px;padding:18px;margin-bottom:12px}
.ctitle{font-size:12px;font-weight:600;color:#ccc;margin-bottom:14px}
.brow{display:flex;align-items:center;gap:9px;margin-bottom:6px}
.blabel{font-size:11px;color:#666;width:90px;text-align:right;flex-shrink:0}
.btrack{flex:1;background:#1e1e1e;border-radius:3px;height:18px;overflow:hidden}
.bfill{height:100%;border-radius:3px;display:flex;align-items:center;padding-left:6px;font-size:10px;color:#0c0c0c;font-weight:700}
.bval{font-size:10px;color:#555;width:34px;flex-shrink:0}
.hlist{display:flex;flex-direction:column;gap:8px}
.hitem{background:#141414;border:1px solid #1e1e1e;border-radius:9px;padding:12px 14px;display:flex;align-items:center;gap:12px}
.hinfo{flex:1}
.htitle{font-size:13px;font-weight:600;color:#f0ece4;margin-bottom:2px}
.hmeta{font-size:11px;color:#555}
.htime{font-size:10px;color:#3a3a3a;white-space:nowrap;flex-shrink:0}
.tag{display:inline-block;padding:3px 8px;border-radius:20px;font-size:10px;font-weight:700;margin-right:4px}
.tw{background:#0f1f0f;color:#5a9e5a;border:1px solid #1a3a1a}
.ts{background:#0f0f1f;color:#5a7ace;border:1px solid #1a1a3a}
.tv{background:#1a1a0f;color:#b8b85a;border:1px solid #2a2a1a}
.pgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{background:#141414;border:1px solid #1e1e1e;border-radius:11px;padding:20px}
.ptitle{font-size:13px;font-weight:700;color:#ccc;margin-bottom:14px}
.srow2{display:flex;flex-direction:column;gap:5px;margin-bottom:12px}
.shdr2{display:flex;justify-content:space-between}
.slabel{font-size:11px;color:#666}
.sval{font-size:11px;color:#e0a84b;font-weight:700}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:#242424;border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:#e0a84b;cursor:pointer}
.rbox{background:#1a1600;border:1px solid #2e2200;border-radius:9px;padding:16px;text-align:center;margin-top:14px}
.rlabel{font-size:11px;color:#777;margin-bottom:4px}
.rval{font-family:'Bebas Neue',sans-serif;font-size:36px;color:#e0a84b;letter-spacing:1px}
.pgrid2{display:grid;grid-template-columns:280px 1fr;gap:16px}
.pcard{background:#141414;border:1px solid #1e1e1e;border-radius:12px;padding:24px;text-align:center}
.pavatar{width:72px;height:72px;border-radius:50%;background:#e0a84b;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:800;color:#0c0c0c;margin:0 auto 12px}
.pname{font-family:'Bebas Neue',sans-serif;font-size:22px;color:#f0ece4;letter-spacing:1px;margin-bottom:3px}
.pemail{font-size:11px;color:#555;margin-bottom:16px}
.pstats{display:flex;justify-content:space-around;margin-bottom:16px}
.pstat-v{font-size:20px;font-weight:800;color:#e0a84b}
.pstat-l{font-size:10px;color:#555}
.ttags{display:flex;flex-wrap:wrap;gap:5px;justify-content:center}
.ttag{padding:4px 9px;border-radius:20px;font-size:10px;background:#1a1600;border:1px solid #2e2200;color:#e0a84b}
.empty{color:#333;font-size:13px;text-align:center;padding:50px 0}
.srow-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.sh{font-size:15px;font-weight:700;color:#ccc}
.sbanner{background:#1a1a1a;border:1px solid #242424;border-left:3px solid #e0a84b;border-radius:7px;padding:9px 13px;font-size:12px;color:#ccc;margin-bottom:12px}
.erow{display:flex;justify-content:flex-end;margin-bottom:10px}
.int-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:18px}
.int-pill{padding:9px 6px;border-radius:8px;font-size:12px;border:1px solid #2a2a2a;background:#141414;color:#777;cursor:pointer;transition:all .15s;text-align:center;font-weight:500}
.int-pill:hover{border-color:#e0a84b;color:#e0a84b}
.int-pill.sel{background:#1a1600;border-color:#e0a84b;color:#e0a84b}
.auth-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;background:radial-gradient(ellipse at 30% 40%,#1c1000,#0c0c0c 65%);position:relative;overflow:hidden}
.auth-bg{position:absolute;font-family:'Bebas Neue',sans-serif;font-size:260px;color:rgba(255,255,255,.025);user-select:none;pointer-events:none;top:50%;left:50%;transform:translate(-50%,-50%);white-space:nowrap;letter-spacing:6px}
.auth-card{background:#141414;border:1px solid #242424;border-radius:20px;padding:44px 42px;width:420px;position:relative;z-index:2;box-shadow:0 50px 100px rgba(0,0,0,.7)}
.auth-logo{font-family:'Bebas Neue',sans-serif;font-size:28px;color:#e0a84b;letter-spacing:2px;margin-bottom:4px}
.auth-tag{font-size:12px;color:#555;margin-bottom:32px;line-height:1.6}
.auth-tabs{display:flex;border-bottom:1px solid #222;margin-bottom:26px}
.auth-tab{flex:1;text-align:center;padding:10px;font-size:13px;font-weight:500;color:#666;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.auth-tab.active{color:#e0a84b;border-bottom-color:#e0a84b}
.fields{display:flex;flex-direction:column;gap:13px;margin-bottom:18px}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:11px;color:#777;letter-spacing:.4px}
.field input{background:#1c1c1c;border:1px solid #2a2a2a;border-radius:8px;padding:11px 14px;color:#f0ece4;font-family:'DM Sans',sans-serif;font-size:14px;outline:none;transition:border-color .2s}
.field input:focus{border-color:#e0a84b}
.field input::placeholder{color:#3a3a3a}
.btn-main{width:100%;background:#e0a84b;color:#0c0c0c;border:none;border-radius:10px;padding:13px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:14px;cursor:pointer;transition:background .15s;margin-bottom:12px}
.btn-main:hover{background:#f0b95c}
.btn-main:disabled{opacity:.4;cursor:not-allowed}
.or-div{display:flex;align-items:center;gap:10px;font-size:11px;color:#3a3a3a;margin-bottom:12px}
.or-div::before,.or-div::after{content:'';flex:1;height:1px;background:#222}
.btn-google{width:100%;background:#1c1c1c;border:1px solid #2e2e2e;border-radius:10px;padding:12px;display:flex;align-items:center;justify-content:center;gap:10px;font-family:'DM Sans',sans-serif;font-size:13px;color:#d0ccc5;cursor:pointer;transition:all .15s;font-weight:500}
.btn-google:hover{border-color:#e0a84b;color:#e0a84b;background:#1a1600}
.auth-foot{font-size:12px;color:#555;text-align:center;margin-top:16px}
.auth-foot span{color:#e0a84b;cursor:pointer}
.code-inputs{display:flex;gap:8px;justify-content:center;margin-bottom:18px}
.code-digit{width:48px;height:56px;background:#1c1c1c;border:1px solid #2a2a2a;border-radius:10px;font-size:22px;font-weight:700;color:#e0a84b;text-align:center;outline:none;transition:border-color .2s;font-family:'DM Sans',sans-serif}
.code-digit:focus{border-color:#e0a84b}
.code-demo{background:#1c1c1c;border:1px solid #2a2a2a;border-radius:8px;padding:10px 14px;font-size:12px;color:#888;margin-bottom:18px;text-align:center}
.code-demo strong{color:#e0a84b;font-size:22px;letter-spacing:6px;display:block;margin-top:4px}
`;

/* ═══════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════ */
export default function App() {
  const [step, setStep]         = useState("login");
  const [authTab, setAuthTab]   = useState("login");
  const [user, setUser]         = useState(null);
  const [interests, setInterests] = useState([]);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPass, setLoginPass]   = useState("");
  const [signupName, setSignupName] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPass, setSignupPass]   = useState("");
  const [verifyCode, setVerifyCode]   = useState(["","","","","",""]);
  const [demoCode, setDemoCode]       = useState("");
  const [codeError, setCodeError]     = useState("");
  const digitRefs = useRef([]);

  const [page, setPage]         = useState("For You");
  const [activity, setActivity] = useState([]);
  const [watched, setWatched]   = useState([]);
  const [query, setQuery]       = useState("");
  const [selGenres, setSelGenres] = useState([]);
  const [searched, setSearched]   = useState(false);
  const [histTab, setHistTab]     = useState("All");
  const [explTab, setExplTab]     = useState("Genres");
  const [budget, setBudget]       = useState(50);
  const [pop, setPop]             = useState(50);
  const [runtime, setRuntime]     = useState(120);
  const [vrating, setVrating]     = useState(7.0);
  const [votes, setVotes]         = useState(5000);
  const [showRev, setShowRev]     = useState(false);
  const [selectedMovie, setSelectedMovie] = useState(null);

  const logActivity = (action, movie) =>
    setActivity(p=>[{title:movie.title,poster:movie.poster,genres:movie.genres,action,time:ts()},...p]);

  const markWatched = (movie) => {
    if(!watched.includes(movie.id)){
      setWatched(p=>[...p,movie.id]);
      logActivity("Watched",movie);
    }
  };

  const openMovie = (movie) => {
    logActivity("Viewed",movie);
    setSelectedMovie(movie);
  };

  const toggleInterest = g => setInterests(p=>p.includes(g)?p.filter(x=>x!==g):[...p,g]);
  const toggleGenre    = g => setSelGenres(p=>p.includes(g)?p.filter(x=>x!==g):[...p,g]);

  const getPersonalized = () => {
    const gc={};
    [...interests,...activity.flatMap(a=>a.genres)].forEach(g=>{gc[g]=(gc[g]||0)+1});
    return ALL_MOVIES.filter(m=>!watched.includes(m.id))
      .map(m=>({...m,score:m.genres.reduce((s,g)=>s+(gc[g]||0),0)+m.rating*.4}))
      .sort((a,b)=>b.score-a.score);
  };

  const filtered = searched
    ? ALL_MOVIES.filter(m=>m.title.toLowerCase().includes(query.toLowerCase())||(selGenres.length===0||m.genres.some(g=>selGenres.includes(g))))
    : [];

  const handleDigit=(i,val)=>{
    if(!/^\d*$/.test(val))return;
    const n=[...verifyCode];n[i]=val.slice(-1);setVerifyCode(n);setCodeError("");
    if(val&&i<5)digitRefs.current[i+1]?.focus();
  };
  const handleKey=(i,e)=>{if(e.key==="Backspace"&&!verifyCode[i]&&i>0)digitRefs.current[i-1]?.focus()};

  const doLoginEmail=()=>{
    if(!loginEmail||!loginPass)return;
    const nm=loginEmail.split("@")[0].replace(/[._]/g," ").replace(/\b\w/g,c=>c.toUpperCase());
    setUser({name:nm,email:loginEmail,avatar:nm.slice(0,2).toUpperCase(),provider:"email"});
    setStep("app");
  };
  const doSignup=()=>{
    if(!signupName||!signupEmail||!signupPass)return;
    const c=generateCode();setDemoCode(c);
    setUser({name:signupName,email:signupEmail,avatar:signupName.slice(0,2).toUpperCase(),provider:"email"});
    setStep("verify");
  };
  const doGoogle=()=>{
    setUser({name:"Ali Hassan",email:"ali@gmail.com",avatar:"AH",provider:"google"});
    if(authTab==="signup"){const c=generateCode();setDemoCode(c);setStep("verify");}
    else setStep("app");
  };
  const doVerify=()=>{
    if(verifyCode.join("")===demoCode)setStep("interests");
    else setCodeError("Incorrect code. Try again.");
  };

  const personalized = getPersonalized();
  const estimated    = estimateRevenue(budget,pop,runtime,vrating,votes);
  const PAGES = ["For You","Recommendations","History","Explore Data","Revenue Predictor","Profile"];
  const ICONS = {"For You":"✨","Recommendations":"🎬","History":"🕐","Explore Data":"📊","Revenue Predictor":"💰","Profile":"👤"};

  /* AUTH SCREENS */
  if(step!=="app") return (
    <>
      <style>{CSS}</style>
      <div className="auth-wrap">
        <div className="auth-bg">CINEMA</div>
        {step==="verify"&&(
          <div className="auth-card su">
            <div style={{fontSize:36,textAlign:"center",marginBottom:10}}>✉️</div>
            <div style={{fontSize:18,fontWeight:700,textAlign:"center",marginBottom:6}}>Check your email</div>
            <div style={{fontSize:13,color:"#666",textAlign:"center",marginBottom:18,lineHeight:1.6}}>
              We sent a 6-digit code to <strong style={{color:"#f0ece4"}}>{user?.email}</strong>
            </div>
            <div className="code-demo">Demo code (simulating email delivery):<strong>{demoCode}</strong></div>
            <div className="code-inputs">
              {verifyCode.map((d,i)=>(
                <input key={i} className="code-digit" maxLength={1} value={d}
                  onChange={e=>handleDigit(i,e.target.value)} onKeyDown={e=>handleKey(i,e)}
                  ref={el=>digitRefs.current[i]=el} autoFocus={i===0}/>
              ))}
            </div>
            {codeError&&<div style={{fontSize:12,color:"#e07b5b",textAlign:"center",marginBottom:10}}>{codeError}</div>}
            <button className="btn-main" onClick={doVerify}>Verify & Continue</button>
            <div style={{fontSize:12,color:"#555",textAlign:"center"}}>
              Didn't get it? <span style={{color:"#e0a84b",cursor:"pointer"}} onClick={()=>{const c=generateCode();setDemoCode(c);setVerifyCode(["","","","","",""]);setCodeError("");}}>Resend</span>
            </div>
          </div>
        )}
        {step==="interests"&&(
          <div className="auth-card su">
            <div style={{fontSize:11,color:"#555",letterSpacing:.5,marginBottom:5}}>STEP 2 OF 2</div>
            <div style={{fontSize:17,fontWeight:700,color:"#f0ece4",marginBottom:5}}>What do you love watching?</div>
            <div style={{fontSize:13,color:"#666",marginBottom:20,lineHeight:1.5}}>Pick genres to personalise your recommendations.</div>
            <div className="int-grid">{INTERESTS.map(g=><div key={g} className={`int-pill ${interests.includes(g)?"sel":""}`} onClick={()=>toggleInterest(g)}>{g}</div>)}</div>
            <button className="btn-main" onClick={()=>setStep("app")} disabled={interests.length===0} style={{opacity:interests.length===0?.4:1}}>
              {interests.length===0?"Select at least one":`Continue with ${interests.length} genre${interests.length>1?"s":""}`}
            </button>
            <div className="auth-foot"><span onClick={()=>setStep("app")}>Skip for now →</span></div>
          </div>
        )}
        {(step==="login"||step==="signup")&&(
          <div className="auth-card su">
            <div className="auth-logo">Cinema to Watch</div>
            <div className="auth-tag">Movie recommendation engine powered by Machine Learning</div>
            <div className="auth-tabs">
              {["login","signup"].map(t=>(
                <div key={t} className={`auth-tab ${authTab===t?"active":""}`} onClick={()=>{setAuthTab(t);setStep(t);}}>
                  {t==="login"?"Sign In":"Create Account"}
                </div>
              ))}
            </div>
            {authTab==="login"?(
              <>
                <div className="fields">
                  <div className="field"><label>Email</label><input type="email" placeholder="you@example.com" value={loginEmail} onChange={e=>setLoginEmail(e.target.value)}/></div>
                  <div className="field"><label>Password</label><input type="password" placeholder="••••••••" value={loginPass} onChange={e=>setLoginPass(e.target.value)}/></div>
                </div>
                <button className="btn-main" onClick={doLoginEmail}>Sign In</button>
                <div className="or-div">or</div>
                <button className="btn-google" onClick={doGoogle}><GoogleIcon/>Continue with Google</button>
                <div className="auth-foot">No account? <span onClick={()=>{setAuthTab("signup");setStep("signup")}}>Sign up</span></div>
              </>
            ):(
              <>
                <div className="fields">
                  <div className="field"><label>Full name</label><input type="text" placeholder="Your name" value={signupName} onChange={e=>setSignupName(e.target.value)}/></div>
                  <div className="field"><label>Email</label><input type="email" placeholder="you@example.com" value={signupEmail} onChange={e=>setSignupEmail(e.target.value)}/></div>
                  <div className="field"><label>Password (min. 8 chars)</label><input type="password" placeholder="••••••••" value={signupPass} onChange={e=>setSignupPass(e.target.value)}/></div>
                </div>
                <button className="btn-main" onClick={doSignup}>Create Account & Verify</button>
                <div className="or-div">or</div>
                <button className="btn-google" onClick={doGoogle}><GoogleIcon/>Sign up with Google</button>
                <div className="auth-foot">Have account? <span onClick={()=>{setAuthTab("login");setStep("login")}}>Sign in</span></div>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );

  /* MAIN APP */
  return (
    <>
      <style>{CSS}</style>

      {selectedMovie&&(
        <MovieModal
          movie={selectedMovie}
          onClose={()=>setSelectedMovie(null)}
          watched={watched}
          onMarkWatched={markWatched}
          onOpenSimilar={(m)=>{ logActivity("Viewed",m); setSelectedMovie(m); }}
        />
      )}

      <div className="app">
        <div className="sidebar">
          <div className="s-logo">Cinema to Watch</div>
          <div className="s-sub">Movie recommendation engine<br/>powered by Machine Learning</div>
          <div className="s-label">Navigation</div>
          {PAGES.map(p=>(
            <div key={p} className={`nav ${page===p?"active":""}`} onClick={()=>{setPage(p);setSearched(false);}}>
              {ICONS[p]} {p}
            </div>
          ))}
          <hr className="s-hr"/>
          <div className="user-chip">
            <div className="avatar">{user?.avatar}</div>
            <div><div className="u-name">{user?.name}</div><div className="u-email">{user?.email}</div></div>
          </div>
          <div className="signout" onClick={()=>{setUser(null);setStep("login");setAuthTab("login");setActivity([]);setWatched([]);setInterests([]);setPage("For You");}}>Sign out</div>
        </div>

        <div className="main">

          {/* FOR YOU */}
          {page==="For You"&&(
            <div className="fi">
              <div className="pg-title">For You, {user?.name?.split(" ")[0]}</div>
              <div className="pg-sub">Personalised picks based on your activity. Click any movie for full details.</div>
              <div className="hero">
                <div style={{fontSize:28}}>🧠</div>
                <div>
                  <h3>Your Taste Profile</h3>
                  <p>{activity.length===0&&interests.length===0?"Start watching or clicking movies to build your profile."
                    :`Based on ${activity.length} activit${activity.length===1?"y":"ies"} · Interests: ${interests.slice(0,4).join(", ")||"None set"}`}</p>
                </div>
              </div>
              <div className="srow-hdr"><div className="sh">Recommended For You</div><div style={{fontSize:11,color:"#444"}}>Click any poster for full details</div></div>
              {personalized.length===0?<div className="empty">You've seen everything!</div>
                :<div className="mgrid">{personalized.slice(0,10).map((m,i)=><MovieCard key={m.id} movie={m} onOpen={openMovie} watched={watched} onMarkWatched={markWatched} delay={i*.04}/>)}</div>
              }
              {watched.length>0&&(<>
                <div className="srow-hdr" style={{marginTop:24}}><div className="sh">Recently Watched</div></div>
                <div className="hlist">
                  {activity.filter(a=>a.action==="Watched").slice(0,3).map((a,i)=>(
                    <div className="hitem" key={i}>
                      <Poster src={a.poster} alt={a.title} style={{width:40,height:60,borderRadius:5,flexShrink:0}}/>
                      <div className="hinfo"><div className="htitle">{a.title}</div><div className="hmeta">{a.genres?.slice(0,2).join(", ")}</div></div>
                      <span className="tag tw">Watched</span><div className="htime">{a.time}</div>
                    </div>
                  ))}
                </div>
              </>)}
              {watched.length===0&&activity.length===0&&<div className="empty" style={{paddingTop:20}}>No activity yet. Click a movie to get started.</div>}
            </div>
          )}

          {/* RECOMMENDATIONS */}
          {page==="Recommendations"&&(
            <div className="fi">
              <div className="pg-title">Recommendations</div>
              <div className="pg-sub">Search for a movie. Click any result to view full details.</div>
              <div className="sbar">
                <div className="ig"><div className="ilabel">Movie title</div>
                  <input className="input" placeholder="e.g. Inception, Parasite, Dune..." value={query}
                    onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==="Enter"&&setSearched(true)}/>
                </div>
                <button className="btn" onClick={()=>{const m=ALL_MOVIES.find(x=>x.title.toLowerCase().includes(query.toLowerCase()));if(m)logActivity("Searched",m);setSearched(true);}}>Search</button>
              </div>
              <div className="ilabel" style={{marginBottom:7}}>Filter by genre</div>
              <div className="gpills">{GENRES.map(g=><div key={g} className={`gpill ${selGenres.includes(g)?"sel":""}`} onClick={()=>toggleGenre(g)}>{g}</div>)}</div>
              {searched?(
                <div className="fi">
                  <div className="sbanner">Results for: <strong>"{query}"</strong> · {filtered.length} found</div>
                  <div className="erow"><button className="btn-ol">⬇ Export CSV</button></div>
                  <div className="mgrid">{filtered.map((m,i)=><MovieCard key={m.id} movie={m} onOpen={openMovie} watched={watched} onMarkWatched={markWatched} delay={i*.04}/>)}</div>
                  {filtered.length===0&&<div className="empty">No results. Try a different title or genre.</div>}
                </div>
              ):<div className="empty">Search for a movie above to get started.</div>}
            </div>
          )}

          {/* HISTORY */}
          {page==="History"&&(
            <div className="fi">
              <div className="pg-title">Activity & History</div>
              <div className="pg-sub">Your real-time activity log.</div>
              <div className="metrics" style={{gridTemplateColumns:"repeat(3,1fr)"}}>
                <div className="mc"><div className="mv">{activity.filter(a=>a.action==="Watched").length}</div><div className="ml">Watched</div></div>
                <div className="mc"><div className="mv">{activity.filter(a=>a.action==="Searched").length}</div><div className="ml">Searched</div></div>
                <div className="mc"><div className="mv">{activity.length}</div><div className="ml">Total</div></div>
              </div>
              <div className="tabs">
                {["All","Watched","Searched","Viewed"].map(t=>(
                  <div key={t} className={`tab ${histTab===t?"active":""}`} onClick={()=>setHistTab(t)}>{t}</div>
                ))}
              </div>
              {activity.filter(a=>histTab==="All"||a.action===histTab).length===0
                ?<div className="empty">No {histTab.toLowerCase()} activity yet.</div>
                :<div className="hlist">
                  {activity.filter(a=>histTab==="All"||a.action===histTab).map((a,i)=>(
                    <div className="hitem fi" key={i} style={{animationDelay:`${i*.03}s`}}>
                      <Poster src={a.poster} alt={a.title} style={{width:40,height:60,borderRadius:5,flexShrink:0}}/>
                      <div className="hinfo"><div className="htitle">{a.title}</div><div className="hmeta">{a.genres?.slice(0,2).join(", ")}</div></div>
                      <span className={`tag ${a.action==="Watched"?"tw":a.action==="Viewed"?"tv":"ts"}`}>{a.action}</span>
                      <div className="htime">{a.time}</div>
                    </div>
                  ))}
                </div>
              }
            </div>
          )}

          {/* EXPLORE DATA */}
          {page==="Explore Data"&&(
            <div className="fi">
              <div className="pg-title">Explore Dataset</div>
              <div className="pg-sub">Visual insights from 4,803 TMDB movies.</div>
              <div className="metrics" style={{gridTemplateColumns:"repeat(4,1fr)"}}>
                {[{v:"4,803",l:"Movies"},{v:"20",l:"Genres"},{v:"6.9",l:"Avg Rating"},{v:"1916–2017",l:"Years"}].map((m,i)=>(
                  <div className="mc" key={i}><div className="mv" style={{fontSize:m.v.length>5?20:28}}>{m.v}</div><div className="ml">{m.l}</div></div>
                ))}
              </div>
              <div className="tabs">
                {["Genres","Ratings","Over Time","Clusters"].map(t=>(
                  <div key={t} className={`tab ${explTab===t?"active":""}`} onClick={()=>setExplTab(t)}>{t}</div>
                ))}
              </div>
              {explTab==="Genres"&&<div className="fi chart">
                <div className="ctitle">Top Genres by Movie Count</div>
                {GENRE_STATS.map((g,i)=>(
                  <div className="brow" key={i}>
                    <div className="blabel">{g.label}</div>
                    <div className="btrack"><div className="bfill" style={{width:`${g.pct}%`,background:"#e0a84b"}}>{g.count}</div></div>
                    <div className="bval">{g.pct}%</div>
                  </div>
                ))}
              </div>}
              {explTab==="Ratings"&&<div className="fi" style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                <div className="chart"><div className="ctitle">Rating Distribution</div>
                  <div style={{display:"flex",alignItems:"flex-end",gap:3,height:100}}>
                    {[2,8,18,42,85,120,145,130,90,45,20,8].map((h,i)=><div key={i} style={{flex:1,height:`${(h/145)*100}%`,background:"#e0a84b",borderRadius:"2px 2px 0 0",opacity:.85}}/>)}
                  </div>
                  <div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:10,color:"#444"}}><span>1.0</span><span>5.5</span><span>10</span></div>
                </div>
                <div className="chart"><div className="ctitle">Runtime Distribution</div>
                  <div style={{display:"flex",alignItems:"flex-end",gap:3,height:100}}>
                    {[5,15,40,95,145,130,90,55,30,18,10,4].map((h,i)=><div key={i} style={{flex:1,height:`${(h/145)*100}%`,background:"#5b8fd4",borderRadius:"2px 2px 0 0",opacity:.85}}/>)}
                  </div>
                  <div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:10,color:"#444"}}><span>30m</span><span>120m</span><span>240m</span></div>
                </div>
              </div>}
              {explTab==="Over Time"&&<div className="fi chart"><div className="ctitle">Movies Released Per Year (1990–2017)</div>
                <div style={{display:"flex",alignItems:"flex-end",gap:2,height:110}}>
                  {[18,22,28,32,38,42,50,55,60,70,85,95,110,125,140,155,165,170,175,180,185,190,195,200,210,215,220,225].map((h,i)=>(
                    <div key={i} style={{flex:1,height:`${(h/225)*100}%`,background:`hsl(${38+i*1.5},70%,${45+i*.3}%)`,borderRadius:"2px 2px 0 0"}}/>
                  ))}
                </div>
                <div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:10,color:"#444"}}><span>1990</span><span>2000</span><span>2010</span><span>2017</span></div>
              </div>}
              {explTab==="Clusters"&&<div className="fi chart"><div className="ctitle">PCA — 2D Cluster Projection</div>
                <svg width="100%" viewBox="0 0 500 270">
                  <rect width="500" height="270" fill="#111" rx="8"/>
                  {[{cx:120,cy:75,r:50,c:"#e0a84b",l:"Blockbusters"},{cx:275,cy:60,r:42,c:"#5b8fd4",l:"Indie"},{cx:400,cy:105,r:38,c:"#7ecb8a",l:"Animated"},{cx:175,cy:190,r:48,c:"#c97ecc",l:"Drama"},{cx:335,cy:195,r:36,c:"#e07b5b",l:"Thriller"},{cx:70,cy:205,r:28,c:"#5bd4c9",l:"Docs"},{cx:455,cy:235,r:24,c:"#d4d45b",l:"Western"}].map((c,i)=>(
                    <g key={i}>
                      {Array.from({length:22}).map((_,j)=><circle key={j} cx={c.cx+(Math.random()-.5)*c.r*2.1} cy={c.cy+(Math.random()-.5)*c.r*2.1} r={2.5} fill={c.c} opacity={.55}/>)}
                      <text x={c.cx} y={c.cy} textAnchor="middle" fill={c.c} fontSize="9" fontFamily="DM Sans" fontWeight="700">{c.l}</text>
                    </g>
                  ))}
                </svg>
              </div>}
            </div>
          )}

          {/* REVENUE PREDICTOR */}
          {page==="Revenue Predictor"&&(
            <div className="fi">
              <div className="pg-title">Revenue Predictor</div>
              <div className="pg-sub">Estimate box office revenue using Random Forest.</div>
              <div className="metrics" style={{gridTemplateColumns:"repeat(3,1fr)",marginBottom:18}}>
                <div className="mc"><div className="mv">0.72</div><div className="ml">R² Score</div></div>
                <div className="mc"><div className="mv">$41M</div><div className="ml">Mean Abs. Error</div></div>
                <div className="mc"><div className="mv">RF</div><div className="ml">Model</div></div>
              </div>
              <div className="pgrid">
                <div className="panel">
                  <div className="ptitle">Feature Importance</div>
                  {[{l:"Budget",p:82,c:"#e0a84b"},{l:"Popularity",p:61,c:"#5b8fd4"},{l:"Vote Count",p:47,c:"#7ecb8a"},{l:"Runtime",p:28,c:"#c97ecc"},{l:"Vote Avg",p:22,c:"#e07b5b"}].map((f,i)=>(
                    <div className="brow" key={i}><div className="blabel">{f.l}</div><div className="btrack"><div className="bfill" style={{width:`${f.p}%`,background:f.c}}>{f.p}%</div></div></div>
                  ))}
                </div>
                <div className="panel">
                  <div className="ptitle">Predict Revenue</div>
                  {[{l:"Budget",v:budget,set:setBudget,min:1,max:300,d:`$${budget}M`},{l:"Popularity",v:pop,set:setPop,min:1,max:300,d:pop},{l:"Runtime (min)",v:runtime,set:setRuntime,min:60,max:240,d:`${runtime}m`},{l:"Vote Average",v:vrating*10,set:v=>setVrating(v/10),min:10,max:100,d:vrating.toFixed(1)},{l:"Vote Count",v:votes/100,set:v=>setVotes(v*100),min:1,max:200,d:`${(votes/1000).toFixed(1)}k`}].map((s,i)=>(
                    <div className="srow2" key={i}>
                      <div className="shdr2"><div className="slabel">{s.l}</div><div className="sval">{s.d}</div></div>
                      <input type="range" min={s.min} max={s.max} value={s.v} onChange={e=>{s.set(Number(e.target.value));setShowRev(false)}}/>
                    </div>
                  ))}
                  <button className="btn" style={{width:"100%",marginTop:12}} onClick={()=>setShowRev(true)}>Predict Revenue</button>
                  {showRev&&<div className="rbox fi"><div className="rlabel">Estimated Box Office Revenue</div><div className="rval">${(estimated/1e6).toFixed(0)}M</div></div>}
                </div>
              </div>
            </div>
          )}

          {/* PROFILE */}
          {page==="Profile"&&(
            <div className="fi">
              <div className="pg-title">Your Profile</div>
              <div className="pg-sub">Account details, taste preferences, and watch history.</div>
              <div className="pgrid2">
                <div>
                  <div className="pcard" style={{marginBottom:12}}>
                    <div className="pavatar">{user?.avatar}</div>
                    <div className="pname">{user?.name}</div>
                    <div className="pemail">{user?.email}</div>
                    <div className="pstats">
                      <div><div className="pstat-v">{watched.length}</div><div className="pstat-l">Watched</div></div>
                      <div><div className="pstat-v">{activity.length}</div><div className="pstat-l">Activities</div></div>
                      <div><div className="pstat-v">{interests.length}</div><div className="pstat-l">Interests</div></div>
                    </div>
                    <div style={{fontSize:11,color:"#444",marginBottom:10}}>via {user?.provider==="google"?"Google":"Email"}</div>
                    {interests.length>0&&<div className="ttags">{interests.map(t=><span className="ttag" key={t}>{t}</span>)}</div>}
                  </div>
                  <div className="panel">
                    <div className="ptitle">Update Interests</div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
                      {INTERESTS.map(g=><div key={g} className={`int-pill ${interests.includes(g)?"sel":""}`} style={{padding:"6px 9px"}} onClick={()=>toggleInterest(g)}>{g}</div>)}
                    </div>
                  </div>
                </div>
                <div>
                  <div className="panel" style={{marginBottom:12}}>
                    <div className="ptitle">Watch History ({watched.length} movies)</div>
                    {watched.length===0
                      ?<div style={{fontSize:12,color:"#444",padding:"10px 0"}}>No movies watched yet.</div>
                      :<div style={{display:"flex",flexDirection:"column",gap:8}}>
                        {ALL_MOVIES.filter(m=>watched.includes(m.id)).map((m,i)=>(
                          <div key={i} onClick={()=>openMovie(m)} style={{display:"flex",alignItems:"center",gap:10,padding:"8px 0",borderBottom:"1px solid #1a1a1a",cursor:"pointer"}}
                            onMouseEnter={e=>e.currentTarget.style.opacity=".75"} onMouseLeave={e=>e.currentTarget.style.opacity="1"}>
                            <Poster src={m.poster} alt={m.title} style={{width:36,height:54,borderRadius:4,flexShrink:0}}/>
                            <div style={{flex:1}}>
                              <div style={{fontSize:12,color:"#d0ccc5",fontWeight:600}}>{m.title}</div>
                              <div style={{fontSize:10,color:"#555"}}>{m.year} · ⭐ {m.rating}</div>
                            </div>
                            <span className="tag tw">✓ Watched</span>
                          </div>
                        ))}
                      </div>
                    }
                  </div>
                  <div className="panel">
                    <div className="ptitle">Account</div>
                    <div className="fields" style={{marginBottom:12}}>
                      <div className="field"><label>Full Name</label><input type="text" defaultValue={user?.name}/></div>
                      <div className="field"><label>Email</label><input type="email" defaultValue={user?.email}/></div>
                    </div>
                    <button className="btn">Save Changes</button>
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </>
  );
}
