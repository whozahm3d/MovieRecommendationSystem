import { useState, useEffect, useRef, useCallback } from "react";

/* ══════════════════════════════════════════════════
   MOVIE DATA
══════════════════════════════════════════════════ */
const MOVIES = [
  { id:1,  title:"Inception",               year:2010, rating:8.3, runtime:148, genres:["Sci-Fi","Thriller"],  director:"Christopher Nolan",  cast:["Leonardo DiCaprio","Joseph Gordon-Levitt","Elliot Page","Tom Hardy"],           overview:"A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",     tagline:"Your mind is the scene of the crime.",              budget:"$160M", revenue:"$836M",  awards:"4 Academy Awards" },
  { id:2,  title:"Interstellar",            year:2014, rating:8.1, runtime:169, genres:["Sci-Fi","Drama"],     director:"Christopher Nolan",  cast:["Matthew McConaughey","Anne Hathaway","Jessica Chastain","Michael Caine"],      overview:"A group of explorers make use of a newly discovered wormhole to surpass the limitations on human space travel.",                                       tagline:"Mankind was born on Earth. It was never meant to die here.", budget:"$165M", revenue:"$773M",  awards:"1 Academy Award" },
  { id:3,  title:"The Dark Knight",         year:2008, rating:9.0, runtime:152, genres:["Action","Crime"],     director:"Christopher Nolan",  cast:["Christian Bale","Heath Ledger","Aaron Eckhart","Gary Oldman"],                 overview:"When the Joker wreaks havoc on Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",  tagline:"Why so serious?",                                   budget:"$185M", revenue:"$1.0B",  awards:"2 Academy Awards" },
  { id:4,  title:"Parasite",                year:2019, rating:8.6, runtime:132, genres:["Drama","Thriller"],   director:"Bong Joon-ho",       cast:["Song Kang-ho","Lee Sun-kyun","Cho Yeo-jeong","Choi Woo-shik"],                overview:"Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",          tagline:"Act like you own the place.",                       budget:"$11M",  revenue:"$258M",  awards:"4 Academy Awards incl. Best Picture" },
  { id:5,  title:"Arrival",                 year:2016, rating:7.9, runtime:116, genres:["Sci-Fi","Mystery"],   director:"Denis Villeneuve",   cast:["Amy Adams","Jeremy Renner","Forest Whitaker","Michael Stuhlbarg"],             overview:"A linguist is recruited by the military to communicate with alien spacecraft appearing around the world.",                                            tagline:"Why are they here?",                                budget:"$47M",  revenue:"$203M",  awards:"1 Academy Award" },
  { id:6,  title:"Blade Runner 2049",       year:2017, rating:8.0, runtime:164, genres:["Sci-Fi","Drama"],     director:"Denis Villeneuve",   cast:["Ryan Gosling","Harrison Ford","Ana de Armas","Sylvia Hoeks"],                 overview:"Young Blade Runner K discovers a long-buried secret that leads him to track down former Blade Runner Rick Deckard, missing for thirty years.",         tagline:"The key to the future is finally unearthed.",       budget:"$150M", revenue:"$260M",  awards:"2 Academy Awards" },
  { id:7,  title:"Whiplash",                year:2014, rating:8.5, runtime:107, genres:["Drama","Music"],      director:"Damien Chazelle",    cast:["Miles Teller","J.K. Simmons","Melissa Benoist","Paul Reiser"],                overview:"A promising young drummer enrolls at a cut-throat music conservatory where his dreams are pursued by an instructor who will stop at nothing.",       tagline:"The road to greatness can take you to the edge.",   budget:"$3.3M", revenue:"$49M",   awards:"3 Academy Awards" },
  { id:8,  title:"The Prestige",            year:2006, rating:8.5, runtime:130, genres:["Mystery","Drama"],    director:"Christopher Nolan",  cast:["Hugh Jackman","Christian Bale","Michael Caine","Scarlett Johansson"],          overview:"After a tragic accident, two stage magicians engage in a competitive battle to create the ultimate illusion while sacrificing everything.",            tagline:"Are you watching closely?",                         budget:"$40M",  revenue:"$109M",  awards:"2 Oscar nominations" },
  { id:9,  title:"Oppenheimer",             year:2023, rating:8.3, runtime:180, genres:["Drama","History"],    director:"Christopher Nolan",  cast:["Cillian Murphy","Emily Blunt","Matt Damon","Robert Downey Jr."],              overview:"The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II.",                      tagline:"The world forever changes.",                        budget:"$100M", revenue:"$952M",  awards:"7 Academy Awards incl. Best Picture" },
  { id:10, title:"Dune",                    year:2021, rating:7.9, runtime:155, genres:["Sci-Fi","Adventure"], director:"Denis Villeneuve",   cast:["Timothée Chalamet","Rebecca Ferguson","Oscar Isaac","Josh Brolin"],           overview:"A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future.", tagline:"Beyond fear, destiny awaits.",                   budget:"$165M", revenue:"$401M",  awards:"6 Academy Awards" },
  { id:11, title:"Joker",                   year:2019, rating:8.4, runtime:122, genres:["Crime","Drama"],      director:"Todd Phillips",      cast:["Joaquin Phoenix","Robert De Niro","Zazie Beetz","Frances Conroy"],             overview:"A mentally troubled comedian embarks on a downward spiral of revolution and bloody crime after being repeatedly failed by society and those around him.", tagline:"Put on a happy face.",                             budget:"$55M",  revenue:"$1.08B", awards:"2 Academy Awards" },
  { id:12, title:"The Shawshank Redemption",year:1994, rating:9.3, runtime:142, genres:["Drama","Crime"],      director:"Frank Darabont",     cast:["Tim Robbins","Morgan Freeman","Bob Gunton","William Sadler"],                 overview:"Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",                               tagline:"Fear can hold you prisoner. Hope can set you free.", budget:"$25M",  revenue:"$16M",   awards:"7 Oscar nominations" },
  { id:13, title:"Fight Club",              year:1999, rating:8.8, runtime:139, genres:["Drama","Thriller"],   director:"David Fincher",      cast:["Brad Pitt","Edward Norton","Helena Bonham Carter","Meat Loaf"],               overview:"An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much, much more.",               tagline:"Mischief. Mayhem. Soap.",                           budget:"$63M",  revenue:"$101M",  awards:"1 Oscar nomination" },
  { id:14, title:"The Martian",             year:2015, rating:7.6, runtime:144, genres:["Sci-Fi","Adventure"], director:"Ridley Scott",       cast:["Matt Damon","Jessica Chastain","Kristen Wiig","Jeff Daniels"],                overview:"An astronaut becomes stranded on Mars after his team assume him dead and must rely on his ingenuity to signal that he is alive.",                        tagline:"Help is only 140 million miles away.",              budget:"$108M", revenue:"$630M",  awards:"7 Oscar nominations" },
  { id:15, title:"Ex Machina",              year:2014, rating:7.7, runtime:108, genres:["Sci-Fi","Mystery"],   director:"Alex Garland",       cast:["Domhnall Gleeson","Alicia Vikander","Oscar Isaac","Sonoya Mizuno"],           overview:"A programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a remarkable A.I.", tagline:"There is nothing more human than the will to survive.", budget:"$15M", revenue:"$36M", awards:"1 Academy Award" },
  { id:16, title:"Gravity",                 year:2013, rating:7.0, runtime:91,  genres:["Sci-Fi","Thriller"],  director:"Alfonso Cuarón",     cast:["Sandra Bullock","George Clooney","Ed Harris"],                                overview:"Two astronauts work together to survive after an accident leaves them stranded in space with no way home.",                                          tagline:"Don't let go.",                                     budget:"$100M", revenue:"$723M",  awards:"7 Academy Awards" },
];

// Known-good Wikipedia poster URLs (fallback, always load)
const WIKI_POSTERS = {
  1:  "https://upload.wikimedia.org/wikipedia/en/2/2e/Inception_%282010%29_theatrical_poster.jpg",
  2:  "https://upload.wikimedia.org/wikipedia/en/b/bc/Interstellar_film_poster.jpg",
  3:  "https://upload.wikimedia.org/wikipedia/en/1/1c/The_Dark_Knight_%282008_film%29.jpg",
  4:  "https://upload.wikimedia.org/wikipedia/en/5/53/Parasite_%282019_film%29.png",
  5:  "https://upload.wikimedia.org/wikipedia/en/e/e0/Arrival_%28film%29_poster.jpg",
  6:  "https://upload.wikimedia.org/wikipedia/en/9/9b/Blade_Runner_2049_poster.png",
  7:  "https://upload.wikimedia.org/wikipedia/en/f/f9/Whiplash_%28film%29.png",
  8:  "https://upload.wikimedia.org/wikipedia/en/0/00/The_Prestige_poster.jpg",
  9:  "https://upload.wikimedia.org/wikipedia/en/4/4a/Oppenheimer_%28film%29.jpg",
  10: "https://upload.wikimedia.org/wikipedia/en/8/8e/Dune_%282021_film%29.jpg",
  11: "https://upload.wikimedia.org/wikipedia/en/e/e1/Joker_%282019_film%29_poster.jpg",
  12: "https://upload.wikimedia.org/wikipedia/en/8/81/ShawshankRedemptionMoviePoster.jpg",
  13: "https://upload.wikimedia.org/wikipedia/en/f/fc/Fight_Club_poster.jpg",
  14: "https://upload.wikimedia.org/wikipedia/en/e/e3/The_Martian_film_poster.jpg",
  15: "https://upload.wikimedia.org/wikipedia/en/b/b8/Ex_machina_uk_film_poster.jpg",
  16: "https://upload.wikimedia.org/wikipedia/en/e/ee/Gravity_Poster.jpg",
};

const GENRE_LIST    = ["Action","Animation","Comedy","Crime","Drama","History","Horror","Mystery","Romance","Sci-Fi","Thriller"];
const INTEREST_LIST = ["Sci-Fi","Drama","Thriller","Comedy","Action","Horror","Romance","Mystery","Animation","Crime","History","Documentary"];
const GENRE_STATS   = [{l:"Drama",p:95,n:2297},{l:"Comedy",p:66,n:1603},{l:"Thriller",p:55,n:1323},{l:"Action",p:48,n:1154},{l:"Romance",p:36,n:860},{l:"Adventure",p:33,n:793},{l:"Crime",p:31,n:760},{l:"Sci-Fi",p:27,n:658}];

const ts        = () => new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});
const genCode   = () => String(Math.floor(100000 + Math.random()*900000));
const estRev    = (b,p,r,ra,v) => Math.max(0, b*2.1e6 + p*800000 + (ra-5)*12e6 + v*800);

/* ══════════════════════════════════════════════════
   SVG ICONS
══════════════════════════════════════════════════ */
const PATHS = {
  home:    "M3 9.5L12 3l9 6.5V20a1 1 0 01-1 1H5a1 1 0 01-1-1V9.5zM9 21V12h6v9",
  sparkle: "M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6L12 2z",
  film:    "M7 4v16M17 4v16M3 8h4m10 0h4M3 16h4m10 0h4M4 4h16a1 1 0 011 1v14a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1z",
  clock:   "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zm0-6v-4l3-3",
  chart:   "M3 3v18h18M7 16l4-4 4 4 4-8",
  money:   "M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6",
  user:    "M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z",
  mail:    "M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2zm0 0l8 9 8-9",
  search:  "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z",
  close:   "M18 6L6 18M6 6l12 12",
  play:    "M5 3l14 9-14 9V3z",
  list:    "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2M9 12h6M9 16h4",
  share:   "M4 12v8a2 2 0 002 2h12a2 2 0 002-2v-8M16 6l-4-4-4 4M12 2v13",
  star:    "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
  fire:    "M12 22c5.523 0 10-4.477 10-10 0-4.5-3-8.5-7-10 1 3-1 6-3 7 0-3-1.5-5.5-4-7 0 5-3 7-3 10a7 7 0 007 10z",
  trending:"M13 7h8m0 0v8m0-8l-8 8-4-4-6 6",
};
const Ico = ({n,size=16,color="currentColor",sw=1.8}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    <path d={PATHS[n]}/>
  </svg>
);

/* ══════════════════════════════════════════════════
   POSTER COMPONENT  – always tries Wikipedia URL first
══════════════════════════════════════════════════ */
const Poster = ({ movieId, title, style={} }) => {
  const src = WIKI_POSTERS[movieId] || "";
  const [ok, setOk] = useState(false);
  const [err, setErr] = useState(false);
  return (
    <div style={{position:"relative",background:"#111",width:"100%",aspectRatio:"2/3",...style}}>
      {!ok && !err && (
        <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",background:"linear-gradient(160deg,#1c1c1c,#252525)"}}>
          <div style={{width:28,height:28,borderRadius:"50%",border:"2px solid #2a2a2a",borderTopColor:"#e0a84b",animation:"spin .8s linear infinite"}}/>
        </div>
      )}
      {err && (
        <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",background:"linear-gradient(160deg,#1c1c1c,#252525)",gap:6,padding:10}}>
          <Ico n="film" size={24} color="#2a2a2a"/>
          <div style={{fontSize:9,color:"#2a2a2a",textAlign:"center",lineHeight:1.4}}>{title}</div>
        </div>
      )}
      {src && !err && (
        <img src={src} alt={title} crossOrigin="anonymous"
          onLoad={()=>setOk(true)} onError={()=>{setErr(true);setOk(false);}}
          style={{position:"absolute",inset:0,width:"100%",height:"100%",objectFit:"cover",display:ok?"block":"none"}}
        />
      )}
    </div>
  );
};

/* ══════════════════════════════════════════════════
   GOOGLE ICON
══════════════════════════════════════════════════ */
const GIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24">
    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
  </svg>
);

/* ══════════════════════════════════════════════════
   MOVIE MODAL
   getPosterUrl is a function (id) => url passed from App
══════════════════════════════════════════════════ */
const MovieModal = ({ movie, watchedIds, onClose, onWatch, similar, onOpenMovie }) => {
  const watched = watchedIds.includes(movie.id);
  const stars   = Math.round(movie.rating / 2);
  return (
    <div style={{position:"fixed",inset:0,zIndex:9999,background:"rgba(0,0,0,.92)",display:"flex",alignItems:"flex-start",justifyContent:"center",overflowY:"auto",padding:"24px 16px"}} onClick={onClose}>
      <div style={{width:"100%",maxWidth:880,background:"#141414",borderRadius:16,overflow:"hidden",border:"1px solid #2a2a2a",boxShadow:"0 40px 100px rgba(0,0,0,.9)",animation:"mIn .3s ease"}} onClick={e=>e.stopPropagation()}>
        {/* HERO BACKDROP */}
        <div style={{position:"relative",height:300,background:"linear-gradient(135deg,#1a1a1a,#111)",display:"flex",alignItems:"center",justifyContent:"center",overflow:"hidden"}}>
          <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",opacity:.12}}>
            <Ico n="film" size={180} color="#e0a84b"/>
          </div>
          <div style={{position:"absolute",inset:0,background:"linear-gradient(to top,#141414 0%,rgba(20,20,20,.3) 60%,transparent 100%)"}}/>
          <button onClick={onClose} style={{position:"absolute",top:14,right:14,width:34,height:34,borderRadius:"50%",background:"rgba(0,0,0,.7)",border:"1px solid #333",color:"#f0ece4",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}}><Ico n="close" size={16}/></button>
          <div style={{position:"absolute",top:14,left:14,background:"#e0a84b",color:"#0c0c0c",padding:"3px 10px",borderRadius:20,fontWeight:800,fontSize:12}}>⭐ {movie.rating}</div>
          <div style={{position:"absolute",bottom:0,left:0,right:0,padding:"0 26px 20px",display:"flex",gap:20,alignItems:"flex-end"}}>
            <div style={{width:90,flexShrink:0,borderRadius:8,overflow:"hidden",border:"2px solid #2a2a2a",boxShadow:"0 8px 30px rgba(0,0,0,.8)"}}>
              <Poster movieId={movie.id} title={movie.title}/>
            </div>
            <div style={{flex:1,paddingBottom:4}}>
              <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:36,color:"#f0ece4",lineHeight:1,letterSpacing:1,marginBottom:5}}>{movie.title}</div>
              <div style={{fontStyle:"italic",color:"#e0a84b",fontSize:12,marginBottom:9}}>{movie.tagline}</div>
              <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                {movie.genres.map(g=><span key={g} style={{padding:"2px 9px",borderRadius:20,fontSize:11,background:"rgba(224,168,75,.15)",border:"1px solid rgba(224,168,75,.3)",color:"#e0a84b",fontWeight:600}}>{g}</span>)}
              </div>
            </div>
          </div>
        </div>
        {/* BODY */}
        <div style={{padding:"22px 26px"}}>
          {/* ACTIONS */}
          <div style={{display:"flex",gap:9,marginBottom:24,flexWrap:"wrap"}}>
            <button onClick={()=>onWatch(movie)} style={{background:watched?"#0f1f0f":"#e0a84b",color:watched?"#5a9e5a":"#0c0c0c",border:watched?"1px solid #1a3a1a":"none",borderRadius:7,padding:"10px 20px",fontWeight:700,fontSize:12,cursor:"pointer",fontFamily:"'DM Sans',sans-serif",display:"flex",alignItems:"center",gap:7}}>
              <Ico n="play" size={14} color={watched?"#5a9e5a":"#0c0c0c"}/>{watched?"✓ Watched":"Mark as Watched"}
            </button>
            <button style={{background:"#1a1a1a",color:"#d0ccc5",border:"1px solid #2a2a2a",borderRadius:7,padding:"10px 20px",fontWeight:700,fontSize:12,cursor:"pointer",fontFamily:"'DM Sans',sans-serif",display:"flex",alignItems:"center",gap:7}}><Ico n="list" size={14} color="#d0ccc5"/>Add to List</button>
            <button style={{background:"#1a1a1a",color:"#d0ccc5",border:"1px solid #2a2a2a",borderRadius:7,padding:"10px 20px",fontWeight:700,fontSize:12,cursor:"pointer",fontFamily:"'DM Sans',sans-serif",display:"flex",alignItems:"center",gap:7}}><Ico n="share" size={14} color="#d0ccc5"/>Share</button>
          </div>
          {/* GRID */}
          <div style={{display:"grid",gridTemplateColumns:"1fr 280px",gap:26,marginBottom:24}}>
            <div>
              <div style={{fontSize:11,fontWeight:700,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:9}}>Overview</div>
              <p style={{fontSize:13,color:"#b0aba4",lineHeight:1.8,marginBottom:22}}>{movie.overview}</p>
              <div style={{fontSize:11,fontWeight:700,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:11}}>Cast</div>
              <div style={{display:"flex",gap:9,flexWrap:"wrap",marginBottom:22}}>
                {movie.cast.map((nm,i)=>(
                  <div key={i} style={{background:"#1e1e1e",border:"1px solid #262626",borderRadius:8,padding:"8px 11px",textAlign:"center",minWidth:70}}>
                    <div style={{width:40,height:40,borderRadius:"50%",background:`hsl(${i*55},28%,24%)`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,fontWeight:700,color:"#d0ccc5",margin:"0 auto 6px",border:"1px solid #333"}}>{nm.split(" ").map(w=>w[0]).join("").slice(0,2)}</div>
                    <div style={{fontSize:10,color:"#d0ccc5",fontWeight:600,lineHeight:1.3}}>{nm.split(" ")[0]}<br/><span style={{color:"#444"}}>{nm.split(" ").slice(1).join(" ")}</span></div>
                  </div>
                ))}
              </div>
              <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:6}}>
                {[1,2,3,4,5].map(s=><span key={s} style={{fontSize:17,color:s<=stars?"#e0a84b":"#252525"}}>★</span>)}
                <span style={{fontSize:13,color:"#e0a84b",fontWeight:700,marginLeft:4}}>{movie.rating}</span>
                <span style={{fontSize:11,color:"#444"}}>/10</span>
              </div>
              {movie.awards && <div style={{fontSize:11,color:"#777",fontStyle:"italic"}}>🏆 {movie.awards}</div>}
            </div>
            <div style={{background:"#1a1a1a",border:"1px solid #222",borderRadius:11,padding:18,alignSelf:"start"}}>
              <div style={{fontSize:12,fontWeight:700,color:"#ccc",marginBottom:14}}>Details</div>
              {[["Director",movie.director],["Year",movie.year],["Runtime",`${Math.floor(movie.runtime/60)}h ${movie.runtime%60}m`],["Budget",movie.budget],["Box Office",movie.revenue],["Status","Released"]].map(([k,v])=>(
                <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"8px 0",borderBottom:"1px solid #1e1e1e",gap:8}}>
                  <span style={{fontSize:11,color:"#555"}}>{k}</span>
                  <span style={{fontSize:11,color:"#d0ccc5",fontWeight:500,textAlign:"right"}}>{v}</span>
                </div>
              ))}
            </div>
          </div>
          {/* SIMILAR */}
          {similar.length > 0 && (
            <div>
              <div style={{fontSize:11,fontWeight:700,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:12}}>More Like This</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:9}}>
                {similar.slice(0,5).map(m=>(
                  <div key={m.id} onClick={()=>onOpenMovie(m)}
                    style={{background:"#1e1e1e",border:"1px solid #222",borderRadius:8,overflow:"hidden",cursor:"pointer",transition:"border-color .2s"}}
                    onMouseEnter={e=>e.currentTarget.style.borderColor="#e0a84b"}
                    onMouseLeave={e=>e.currentTarget.style.borderColor="#222"}>
                    <Poster movieId={m.id} title={m.title}/>
                    <div style={{padding:"7px 8px 9px"}}>
                      <div style={{fontSize:10,fontWeight:600,color:"#f0ece4",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{m.title}</div>
                      <div style={{fontSize:9,color:"#555"}}>⭐ {m.rating}</div>
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
};

/* ══════════════════════════════════════════════════
   MOVIE CARD
══════════════════════════════════════════════════ */
const MovieCard = ({movie, watchedIds, onOpen, onWatch, compact=false}) => {
  const w = watchedIds.includes(movie.id);
  return (
    <div onClick={()=>onOpen(movie)}
      style={{background:"#141414",border:"1px solid #1e1e1e",borderRadius:10,overflow:"hidden",cursor:"pointer",transition:"all .2s",position:"relative"}}
      onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-4px)";e.currentTarget.style.borderColor="#e0a84b";e.currentTarget.style.boxShadow="0 10px 28px rgba(224,168,75,.12)";}}
      onMouseLeave={e=>{e.currentTarget.style.transform="";e.currentTarget.style.borderColor="#1e1e1e";e.currentTarget.style.boxShadow="";}}>
      {w && <div style={{position:"absolute",top:6,right:6,zIndex:2,background:"rgba(8,8,8,.88)",color:"#5a9e5a",border:"1px solid #1a3a1a",fontSize:9,fontWeight:700,padding:"2px 6px",borderRadius:10}}>✓</div>}
      <div style={{position:"relative"}}>
        <Poster movieId={movie.id} title={movie.title}/>
        <div style={{position:"absolute",top:6,left:6,background:"#e0a84b",color:"#0c0c0c",fontSize:8,fontWeight:800,padding:"2px 5px",borderRadius:3}}>HD</div>
      </div>
      <div style={{padding:compact?"7px":"10px"}}>
        <div style={{fontSize:compact?10:11,fontWeight:700,color:"#f0ece4",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",marginBottom:2}}>{movie.title}</div>
        <div style={{fontSize:9,color:"#555"}}>{movie.year} · ⭐ {movie.rating}</div>
        {!compact && (
          <div style={{marginTop:7}} onClick={e=>e.stopPropagation()}>
            <button onClick={()=>onWatch(movie)} style={{background:w?"#0f1f0f":"#1a1600",color:w?"#5a9e5a":"#e0a84b",border:`1px solid ${w?"#1a3a1a":"#2e2200"}`,borderRadius:5,padding:"4px 9px",fontSize:10,fontWeight:700,cursor:"pointer",fontFamily:"'DM Sans',sans-serif"}}>
              {w?"✓ Watched":"+ Watched"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

/* ══════════════════════════════════════════════════
   GLOBAL CSS
══════════════════════════════════════════════════ */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:#0c0c0c;color:#f0ece4;min-height:100vh;overflow-x:hidden}
.app{display:flex;min-height:100vh}
@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
@keyframes mIn{from{opacity:0;transform:translateY(18px) scale(.97)}to{opacity:1;transform:none}}
@keyframes fi{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes su{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes heroShift{0%{transform:scale(1)}100%{transform:scale(1.04)}}
@keyframes slideLeft{from{transform:translateX(60px);opacity:0}to{transform:translateX(0);opacity:1}}
.fi{animation:fi .3s ease forwards}
.su{animation:su .35s ease forwards}
.sb{width:214px;min-width:214px;background:#0e0e0e;border-right:1px solid #181818;padding:20px 12px;display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.sb-logo{font-family:'Bebas Neue',sans-serif;font-size:20px;color:#e0a84b;letter-spacing:2px;margin-bottom:2px}
.sb-sub{font-size:9.5px;color:#3a3a3a;margin-bottom:18px;line-height:1.6}
.sb-lbl{font-size:9px;color:#333;text-transform:uppercase;letter-spacing:1.2px;margin:8px 0 5px 4px}
.nav{padding:8px 10px;border-radius:7px;font-size:12px;color:#666;cursor:pointer;display:flex;align-items:center;gap:9px;transition:all .15s;margin-bottom:1px;white-space:nowrap}
.nav:hover{background:#181818;color:#ccc}
.nav.active{background:#1a1600;color:#e0a84b;font-weight:600}
.sb-hr{border:none;border-top:1px solid #181818;margin:10px 0}
.uchip{display:flex;align-items:center;gap:9px;padding:9px 10px;background:#181818;border-radius:8px;margin-top:auto;border:1px solid #1e1e1e;cursor:pointer}
.av{width:30px;height:30px;border-radius:50%;background:#e0a84b;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;color:#0c0c0c;flex-shrink:0}
.uname{font-size:12px;color:#ccc;font-weight:600;line-height:1.2}
.uemail{font-size:10px;color:#3a3a3a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:130px}
.sout{font-size:10px;color:#3a3a3a;cursor:pointer;text-align:center;margin-top:6px;transition:color .15s;padding:4px}
.sout:hover{color:#e07b5b}
.main{flex:1;overflow-y:auto;max-height:100vh}
.page{padding:26px 28px}
.ptitle{font-family:'Bebas Neue',sans-serif;font-size:26px;color:#f0ece4;letter-spacing:1px;margin-bottom:3px}
.psub{font-size:12px;color:#555;margin-bottom:18px}
.mc{background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:14px;text-align:center}
.mv{font-family:'Bebas Neue',sans-serif;font-size:24px;color:#e0a84b;letter-spacing:1px;line-height:1}
.ml{font-size:10px;color:#555;margin-top:3px}
.hero-bg{position:relative;height:480px;overflow:hidden;background:#0a0a0a}
.hero-grad{position:absolute;inset:0;background:linear-gradient(to right,rgba(12,12,12,.98) 30%,rgba(12,12,12,.4) 70%,rgba(12,12,12,.1) 100%),linear-gradient(to top,#0c0c0c 0%,transparent 40%)}
.hero-content{position:absolute;inset:0;padding:60px 48px;display:flex;flex-direction:column;justify-content:flex-end;max-width:480px}
.scroll-row{display:flex;gap:12px;overflow-x:auto;padding-bottom:8px;scrollbar-width:thin;scrollbar-color:#2a2a2a transparent}
.scroll-row::-webkit-scrollbar{height:3px}
.scroll-row::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:2px}
.row-card{min-width:145px;max-width:145px;flex-shrink:0}
.sbar{display:flex;gap:9px;margin-bottom:11px;align-items:flex-end}
.ig{display:flex;flex-direction:column;gap:4px;flex:1}
.ilbl{font-size:10px;color:#666}
.inp{background:#141414;border:1px solid #222;border-radius:7px;padding:9px 12px;color:#f0ece4;font-family:'DM Sans',sans-serif;font-size:12px;outline:none;transition:border-color .2s;width:100%}
.inp:focus{border-color:#e0a84b}
.inp::placeholder{color:#333}
.btn{background:#e0a84b;color:#0c0c0c;border:none;border-radius:7px;padding:9px 18px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:12px;cursor:pointer;transition:background .15s;height:36px}
.btn:hover{background:#f0b95c}
.btn:disabled{opacity:.4;cursor:not-allowed}
.gpills{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:13px}
.gp{padding:4px 10px;border-radius:20px;font-size:10px;border:1px solid #222;background:#141414;color:#555;cursor:pointer;transition:all .15s}
.gp:hover{border-color:#e0a84b;color:#e0a84b}
.gp.sel{background:#1a1600;border-color:#e0a84b;color:#e0a84b}
.tabs{display:flex;border-bottom:1px solid #1e1e1e;margin-bottom:15px}
.tab{padding:7px 13px;font-size:11px;color:#555;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;font-weight:500}
.tab:hover{color:#ccc}
.tab.active{color:#e0a84b;border-bottom-color:#e0a84b}
.chart{background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:16px;margin-bottom:11px}
.ctitle{font-size:11px;font-weight:600;color:#bbb;margin-bottom:12px}
.brow{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.blbl{font-size:10px;color:#555;width:82px;text-align:right;flex-shrink:0}
.btrack{flex:1;background:#1e1e1e;border-radius:3px;height:16px;overflow:hidden}
.bfill{height:100%;border-radius:3px;display:flex;align-items:center;padding-left:5px;font-size:9px;color:#0c0c0c;font-weight:800}
.hlist{display:flex;flex-direction:column;gap:7px}
.hitem{background:#141414;border:1px solid #1e1e1e;border-radius:8px;padding:10px 12px;display:flex;align-items:center;gap:10px;cursor:pointer;transition:border-color .15s}
.hitem:hover{border-color:#333}
.pgrid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.panel{background:#141414;border:1px solid #1e1e1e;border-radius:10px;padding:17px}
.ptl{font-size:12px;font-weight:700;color:#bbb;margin-bottom:12px}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:#222;border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#e0a84b;cursor:pointer}
.auth-wrap{min-height:100vh;display:flex;align-items:center;justify-content:center;background:radial-gradient(ellipse at 30% 40%,#1c1000 0%,#0c0c0c 65%);position:relative;overflow:hidden}
.auth-bg{position:absolute;font-family:'Bebas Neue',sans-serif;font-size:220px;color:rgba(255,255,255,.018);user-select:none;pointer-events:none;top:50%;left:50%;transform:translate(-50%,-50%);white-space:nowrap;letter-spacing:6px}
.acard{background:#141414;border:1px solid #222;border-radius:16px;padding:40px 38px;width:408px;position:relative;z-index:2;box-shadow:0 50px 100px rgba(0,0,0,.7)}
.alogo{font-family:'Bebas Neue',sans-serif;font-size:24px;color:#e0a84b;letter-spacing:2px;margin-bottom:3px}
.atag{font-size:11px;color:#555;margin-bottom:28px;line-height:1.6}
.atabs{display:flex;border-bottom:1px solid #222;margin-bottom:22px}
.atab{flex:1;text-align:center;padding:9px;font-size:12px;font-weight:500;color:#555;cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.atab.active{color:#e0a84b;border-bottom-color:#e0a84b}
.fields{display:flex;flex-direction:column;gap:11px;margin-bottom:15px}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:10px;color:#666;letter-spacing:.3px}
.field input{background:#1c1c1c;border:1px solid #282828;border-radius:7px;padding:10px 12px;color:#f0ece4;font-family:'DM Sans',sans-serif;font-size:13px;outline:none;transition:border-color .2s}
.field input:focus{border-color:#e0a84b}
.field input::placeholder{color:#383838}
.bmain{width:100%;background:#e0a84b;color:#0c0c0c;border:none;border-radius:8px;padding:11px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:13px;cursor:pointer;transition:background .15s;margin-bottom:10px}
.bmain:hover{background:#f0b95c}
.bmain:disabled{opacity:.4;cursor:not-allowed}
.or-d{display:flex;align-items:center;gap:10px;font-size:10px;color:#333;margin-bottom:10px}
.or-d::before,.or-d::after{content:'';flex:1;height:1px;background:#1e1e1e}
.bgoogle{width:100%;background:#1c1c1c;border:1px solid #282828;border-radius:8px;padding:10px;display:flex;align-items:center;justify-content:center;gap:9px;font-family:'DM Sans',sans-serif;font-size:12px;color:#ccc;cursor:pointer;transition:all .15s;font-weight:500}
.bgoogle:hover{border-color:#e0a84b;color:#e0a84b;background:#1a1600}
.afoot{font-size:11px;color:#555;text-align:center;margin-top:13px}
.afoot span{color:#e0a84b;cursor:pointer}
.code-wrap{display:flex;gap:7px;justify-content:center;margin-bottom:14px}
.cd{width:44px;height:52px;background:#1c1c1c;border:1px solid #282828;border-radius:8px;font-size:20px;font-weight:700;color:#e0a84b;text-align:center;outline:none;transition:border-color .2s;font-family:'DM Sans',sans-serif}
.cd:focus{border-color:#e0a84b}
.intgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:15px}
.intpill{padding:8px 5px;border-radius:7px;font-size:11px;border:1px solid #282828;background:#141414;color:#666;cursor:pointer;transition:all .15s;text-align:center;font-weight:500}
.intpill:hover{border-color:#e0a84b;color:#e0a84b}
.intpill.sel{background:#1a1600;border-color:#e0a84b;color:#e0a84b}
.srow{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.sh{font-size:13px;font-weight:700;color:#bbb}
.sbanner{background:#1a1a1a;border:1px solid #222;border-left:3px solid #e0a84b;border-radius:6px;padding:8px 12px;font-size:12px;color:#ccc;margin-bottom:10px}
.empty{color:#2a2a2a;font-size:12px;text-align:center;padding:36px 0}
.tag{display:inline-block;padding:2px 7px;border-radius:20px;font-size:9px;font-weight:700}
.tw{background:#0f1f0f;color:#5a9e5a;border:1px solid #1a3a1a}
.ts2{background:#0f0f1f;color:#5a7ace;border:1px solid #1a1a3a}
.tv{background:#1a0f1a;color:#9a6ace;border:1px solid #2a1a3a}
.cform input,.cform textarea{background:#141414;border:1px solid #222;border-radius:8px;padding:10px 13px;color:#f0ece4;font-family:'DM Sans',sans-serif;font-size:12px;outline:none;transition:border-color .2s;width:100%;display:block;margin-bottom:10px;resize:vertical}
.cform input:focus,.cform textarea:focus{border-color:#e0a84b}
.cform input::placeholder,.cform textarea::placeholder{color:#333}
.g5{display:grid;grid-template-columns:repeat(5,1fr);gap:11px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:11px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.skel{background:linear-gradient(90deg,#1a1a1a 25%,#222 50%,#1a1a1a 75%);background-size:200% 100%;animation:pulse 1.2s infinite;border-radius:8px}
`;

/* ══════════════════════════════════════════════════
   MAIN APP
══════════════════════════════════════════════════ */
export default function App() {
  // ── auth ──
  const [step, setStep]           = useState("login");
  const [authTab, setAuthTab]     = useState("login");
  const [user, setUser]           = useState(null);
  const [interests, setInterests] = useState([]);
  const [loginEmail, setLE]       = useState("");
  const [loginPass,  setLP]       = useState("");
  const [signupName, setSN]       = useState("");
  const [signupEmail,setSE]       = useState("");
  const [signupPass, setSP]       = useState("");
  const [pendingUser,setPending]  = useState(null);
  // verification
  const [vCode,     setVCode]     = useState(["","","","","",""]);
  const [_code,    _setCode]      = useState("");
  const [vAttempts, setVAttempts] = useState(0);
  const [vLocked,   setVLocked]   = useState(false);
  const [vLockEnd,  setVLockEnd]  = useState(null);
  const [vError,    setVError]    = useState("");
  const [vCountdown,setVCdown]    = useState("");
  const digitRefs = useRef([]);
  // google picker
  const [showGPicker,  setShowGPicker]  = useState(false);
  const [googleMode,   setGoogleMode]   = useState("login");
  // ── app ──
  const [page,      setPage]      = useState("Home");
  const [activity,  setActivity]  = useState([]);
  const [watchedIds,setWatched]   = useState([]);
  const [selMovie,  setSelMovie]  = useState(null);
  const [query,     setQuery]     = useState("");
  const [selGenres, setSelGenres] = useState([]);
  const [searched,  setSearched]  = useState(false);
  const [histTab,   setHistTab]   = useState("All");
  const [exTab,     setExTab]     = useState("Genres");
  const [budget,    setBudget]    = useState(50);
  const [pop,       setPop]       = useState(50);
  const [runtime2,  setRuntime]   = useState(120);
  const [rat2,      setRat]       = useState(7.0);
  const [votes2,    setVotes]     = useState(5000);
  const [showRev,   setShowRev]   = useState(false);
  const [trending,  setTrending]  = useState([]);
  const [trendLoad, setTrendLoad] = useState(false);
  const [contactSent,setContact]  = useState(false);
  const [heroIdx,   setHeroIdx]   = useState(0);
  const [heroKey,   setHeroKey]   = useState(0);
  const [cName,setCName] = useState(""); const [cEmail,setCEmail] = useState(""); const [cMsg,setCMsg] = useState("");

  // ── hero rotation ──
  useEffect(()=>{
    const t = setInterval(()=>{ setHeroIdx(i=>(i+1)%4); setHeroKey(k=>k+1); }, 7000);
    return ()=>clearInterval(t);
  },[]);

  // ── lockout countdown ──
  useEffect(()=>{
    if (!vLocked || !vLockEnd) return;
    const t = setInterval(()=>{
      const rem = vLockEnd - Date.now();
      if (rem <= 0){ setVLocked(false); setVAttempts(0); setVLockEnd(null); setVCdown(""); clearInterval(t); return; }
      const h=Math.floor(rem/3600000), m=Math.floor((rem%3600000)/60000), s=Math.floor((rem%60000)/1000);
      setVCdown(`${h}h ${m}m ${s}s`);
    }, 1000);
    return ()=>clearInterval(t);
  },[vLocked, vLockEnd]);

  // ── trending ──
  const fetchTrending = useCallback(async()=>{
    setTrendLoad(true);
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages",{
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({ model:"claude-sonnet-4-20250514", max_tokens:1200,
          messages:[{role:"user",content:`List 8 trending movies right now (2024-2025). Return ONLY a JSON array. Each item: title, year, rating (1-10), genres (array, 1-2 items from: Action,Comedy,Drama,Horror,Sci-Fi,Thriller,Animation), runtime (minutes), director, overview (max 90 chars). No markdown.`}]
        })
      });
      const data = await res.json();
      const txt = data.content?.find(c=>c.type==="text")?.text||"[]";
      const parsed = JSON.parse(txt.replace(/```json|```/g,"").trim());
      setTrending(Array.isArray(parsed)?parsed.slice(0,8):[]);
    } catch(e){ setTrending([]); }
    setTrendLoad(false);
  },[]);

  useEffect(()=>{ if(step==="app") fetchTrending(); },[step]);

  // ── helpers ──
  const toggleInt  = g => setInterests(p=>p.includes(g)?p.filter(x=>x!==g):[...p,g]);
  const toggleG    = g => setSelGenres(p=>p.includes(g)?p.filter(x=>x!==g):[...p,g]);
  const logAct     = (action,movie) => setActivity(p=>[{action,title:movie.title,mid:movie.id,genres:movie.genres||[],time:ts()},...p]);
  const markWatch  = m => { if(!watchedIds.includes(m.id)){setWatched(p=>[...p,m.id]); logAct("Watched",m);} };
  const openMovie  = m => { setSelMovie(m); logAct("Viewed",m); };
  const getSimilar = m => MOVIES.filter(x=>x.id!==m.id&&x.genres.some(g=>m.genres.includes(g))).sort((a,b)=>b.rating-a.rating);
  const getPersonalized = ()=>{
    const gc={};
    [...interests,...activity.flatMap(a=>a.genres||[])].forEach(g=>{gc[g]=(gc[g]||0)+1;});
    return MOVIES.filter(m=>!watchedIds.includes(m.id)).map(m=>({...m,score:m.genres.reduce((s,g)=>s+(gc[g]||0),0)+m.rating*.4})).sort((a,b)=>b.score-a.score);
  };

  // ── verification ──
  const handleDigit=(i,val)=>{
    if(!/^\d*$/.test(val)) return;
    const n=[...vCode]; n[i]=val.slice(-1); setVCode(n); setVError("");
    if(val&&i<5) digitRefs.current[i+1]?.focus();
  };
  const handleDKey=(i,e)=>{ if(e.key==="Backspace"&&!vCode[i]&&i>0) digitRefs.current[i-1]?.focus(); };
  const sendCode=async(email)=>{
    const code=genCode(); _setCode(code);
    console.log(`[DEV ONLY - not shown to user] Code for ${email}: ${code}`);
    setVCode(["","","","","",""]); setVError(""); setVAttempts(0);
  };
  const doVerify=()=>{
    if(vLocked) return;
    const entered=vCode.join("");
    if(entered.length<6){setVError("Please enter all 6 digits.");return;}
    if(entered===_code){ setUser(pendingUser); setStep("interests"); setVCode(["","","","","",""]); setVAttempts(0); }
    else {
      const na=vAttempts+1; setVAttempts(na);
      if(na>=3){ setVLocked(true); setVLockEnd(Date.now()+4*60*60*1000); setVError("Too many incorrect attempts. Locked for 4 hours."); }
      else { setVError(`Incorrect code. ${3-na} attempt${3-na===1?"":"s"} remaining.`); }
    }
  };

  // ── auth actions ──
  const doLoginEmail=()=>{
    if(!loginEmail||!loginPass) return;
    const nm=loginEmail.split("@")[0].replace(/[._]/g," ").replace(/\b\w/g,c=>c.toUpperCase());
    setUser({name:nm,email:loginEmail,avatar:nm.slice(0,2).toUpperCase(),provider:"email"});
    setStep("app");
  };
  const doSignupEmail=async()=>{
    if(!signupName||!signupEmail||!signupPass||signupPass.length<8) return;
    const u={name:signupName,email:signupEmail,avatar:signupName.slice(0,2).toUpperCase(),provider:"email"};
    setPending(u); await sendCode(signupEmail); setStep("verify");
  };
  const doGoogle=(mode="login")=>{ setGoogleMode(mode); setShowGPicker(true); };
  const doLogout=()=>{
    setUser(null);setStep("login");setAuthTab("login");
    setActivity([]);setWatched([]);setInterests([]);setPage("Home");
    setPending(null);setVAttempts(0);setVLocked(false);setShowGPicker(false);
  };
  const doSearch=()=>{
    if(!query) return;
    const m=MOVIES.find(x=>x.title.toLowerCase().includes(query.toLowerCase()));
    if(m) logAct("Searched",m); setSearched(true);
  };

  const personalized    = getPersonalized();
  const heroMovie       = MOVIES[heroIdx];
  const filteredMovies  = searched
    ? MOVIES.filter(m=>m.title.toLowerCase().includes(query.toLowerCase())||(selGenres.length===0||m.genres.some(g=>selGenres.includes(g))))
    : [];

  // ── Google Picker Modal (inline, no separate component needed) ──
  const DEMO_ACCOUNTS = [
    {name:"Ali Hassan",   email:"ali.hassan@gmail.com",   initials:"AH", color:"#1a73e8"},
    {name:"Ahmed Raza",   email:"ahmed.raza99@gmail.com", initials:"AR", color:"#34a853"},
    {name:"Use another account", email:"",               initials:"+",  color:"#5f6368"},
  ];
  const pickGoogleAccount=(acc)=>{
    if(!acc.email){setShowGPicker(false);return;}
    const u={name:acc.name,email:acc.email,avatar:acc.initials,provider:"google"};
    if(googleMode==="signup"){setPending(u);sendCode(acc.email);setStep("verify");}
    else{setUser(u);setStep("app");}
    setShowGPicker(false);
  };
  const GooglePicker=()=>(
    <div style={{position:"fixed",inset:0,zIndex:10000,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(0,0,0,.7)",backdropFilter:"blur(4px)"}} onClick={()=>setShowGPicker(false)}>
      <div style={{background:"#fff",borderRadius:28,width:360,overflow:"hidden",boxShadow:"0 24px 80px rgba(0,0,0,.5)",animation:"mIn .25s ease"}} onClick={e=>e.stopPropagation()}>
        <div style={{padding:"28px 28px 18px",textAlign:"center",borderBottom:"1px solid #e8eaed"}}>
          <svg width="75" height="24" viewBox="0 0 272 92" style={{marginBottom:16}}>
            <path fill="#EA4335" d="M115.75 47.18c0 12.77-9.99 22.18-22.25 22.18s-22.25-9.41-22.25-22.18C71.25 34.32 81.24 25 93.5 25s22.25 9.32 22.25 22.18zm-9.74 0c0-7.98-5.79-13.44-12.51-13.44S80.99 39.2 80.99 47.18c0 7.9 5.79 13.44 12.51 13.44s12.51-5.55 12.51-13.44z"/>
            <path fill="#FBBC05" d="M163.75 47.18c0 12.77-9.99 22.18-22.25 22.18s-22.25-9.41-22.25-22.18c0-12.85 9.99-22.18 22.25-22.18s22.25 9.32 22.25 22.18zm-9.74 0c0-7.98-5.79-13.44-12.51-13.44s-12.51 5.46-12.51 13.44c0 7.9 5.79 13.44 12.51 13.44s12.51-5.55 12.51-13.44z"/>
            <path fill="#4285F4" d="M209.75 26.34v39.82c0 16.38-9.66 23.07-21.08 23.07-10.75 0-17.22-7.19-19.66-13.07l8.48-3.53c1.51 3.61 5.21 7.87 11.17 7.87 7.31 0 11.84-4.51 11.84-13v-3.19h-.34c-2.18 2.69-6.38 5.04-11.68 5.04-11.09 0-21.25-9.66-21.25-22.09 0-12.52 10.16-22.26 21.25-22.26 5.29 0 9.49 2.35 11.68 4.96h.34v-3.61h9.25zm-8.56 20.92c0-7.81-5.21-13.52-11.84-13.52-6.72 0-12.35 5.71-12.35 13.52 0 7.73 5.63 13.36 12.35 13.36 6.63 0 11.84-5.63 11.84-13.36z"/>
            <path fill="#34A853" d="M225 3v65h-9.5V3h9.5z"/>
            <path fill="#EA4335" d="M262.02 54.48l7.56 5.04c-2.44 3.61-8.32 9.83-18.48 9.83-12.6 0-22.01-9.74-22.01-22.18 0-13.19 9.49-22.18 20.92-22.18 11.51 0 17.14 9.16 18.98 14.11l1.01 2.52-29.65 12.28c2.27 4.45 5.8 6.72 10.75 6.72 4.96 0 8.4-2.44 10.92-6.14zm-23.27-7.98l19.82-8.23c-1.09-2.77-4.37-4.7-8.23-4.7-4.95 0-11.84 4.37-11.59 12.93z"/>
            <path fill="#4285F4" d="M35.29 41.41V32h31.96c.31 1.64.47 3.58.47 5.68 0 7.06-1.93 15.79-8.15 22.01-6.05 6.3-13.78 9.66-24.02 9.66C16.32 69.35.36 53.89.36 34.74.36 15.59 16.32.13 35.55.13c10.5 0 17.98 4.12 23.6 9.49l-6.64 6.64c-4.03-3.78-9.49-6.72-16.97-6.72-13.86 0-24.7 11.17-24.7 25.03 0 13.86 10.84 25.03 24.7 25.03 8.99 0 14.11-3.61 17.39-6.89 2.66-2.66 4.41-6.46 5.1-11.65l-22.74.05z"/>
          </svg>
          <div style={{fontSize:16,fontWeight:500,color:"#202124",marginBottom:6}}>Sign in with Google</div>
          <div style={{fontSize:14,color:"#5f6368"}}>to continue to Cinema to Watch</div>
        </div>
        <div style={{padding:"8px 0"}}>
          {DEMO_ACCOUNTS.map((acc,i)=>(
            <div key={i} onClick={()=>pickGoogleAccount(acc)}
              style={{display:"flex",alignItems:"center",gap:16,padding:"12px 24px",cursor:"pointer",transition:"background .15s"}}
              onMouseEnter={e=>e.currentTarget.style.background="#f8f9fa"}
              onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
              <div style={{width:40,height:40,borderRadius:"50%",background:acc.color,display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",fontWeight:600,fontSize:14,flexShrink:0}}>{acc.initials}</div>
              <div>
                {acc.email?<><div style={{fontSize:14,fontWeight:500,color:"#202124"}}>{acc.name}</div><div style={{fontSize:12,color:"#5f6368"}}>{acc.email}</div></>
                  :<div style={{fontSize:14,color:"#1a73e8",fontWeight:500}}>{acc.name}</div>}
              </div>
            </div>
          ))}
        </div>
        <div style={{padding:"16px 24px",borderTop:"1px solid #e8eaed",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={{fontSize:11,color:"#5f6368"}}>Set <strong>GOOGLE_CLIENT_ID</strong> in .env<br/>to enable real Google Sign-In</span>
          <button onClick={()=>setShowGPicker(false)} style={{background:"none",border:"none",color:"#1a73e8",fontSize:13,fontWeight:500,cursor:"pointer",padding:"6px 12px",borderRadius:4}}>Cancel</button>
        </div>
      </div>
    </div>
  );

  /* ══════════════════════════════════════════
     AUTH SCREENS
  ══════════════════════════════════════════ */
  if (step !== "app") return (
    <>
      <style>{CSS}</style>
      {showGPicker && <GooglePicker/>}
      <div className="auth-wrap">
        <div className="auth-bg">CINEMA</div>
        {/* VERIFY */}
        {step==="verify"&&(
          <div className="acard su">
            <div style={{fontSize:36,textAlign:"center",marginBottom:10}}>✉️</div>
            <div style={{fontSize:17,fontWeight:700,color:"#f0ece4",textAlign:"center",marginBottom:5}}>Check your email</div>
            <div style={{fontSize:12,color:"#666",textAlign:"center",lineHeight:1.6,marginBottom:20}}>
              A 6-digit code was sent to<br/><strong style={{color:"#d0ccc5"}}>{pendingUser?.email}</strong>
            </div>
            {vLocked?(
              <div style={{background:"#1f0f0f",border:"1px solid #3a1a1a",borderRadius:9,padding:16,textAlign:"center",marginBottom:14}}>
                <div style={{color:"#e07b5b",fontWeight:700,fontSize:13,marginBottom:6}}>Account temporarily locked</div>
                <div style={{color:"#666",fontSize:11,marginBottom:8}}>Too many incorrect attempts. Try again after:</div>
                <div style={{color:"#e0a84b",fontFamily:"'Bebas Neue',sans-serif",fontSize:26,letterSpacing:2}}>{vCountdown}</div>
              </div>
            ):(
              <>
                <div className="code-wrap">
                  {vCode.map((d,i)=><input key={i} className="cd" maxLength={1} value={d} onChange={e=>handleDigit(i,e.target.value)} onKeyDown={e=>handleDKey(i,e)} ref={el=>digitRefs.current[i]=el} autoFocus={i===0}/>)}
                </div>
                {vError&&<div style={{fontSize:11,color:"#e07b5b",textAlign:"center",marginBottom:10}}>{vError}</div>}
                <button className="bmain" onClick={doVerify}>Verify & Continue</button>
                <div className="afoot">Didn't receive it? <span onClick={()=>sendCode(pendingUser?.email)}>Resend code</span></div>
              </>
            )}
            <div style={{fontSize:10,color:"#333",textAlign:"center",marginTop:14,borderTop:"1px solid #1e1e1e",paddingTop:12}}>Configure SMTP_EMAIL in .env for real email delivery</div>
          </div>
        )}
        {/* INTERESTS */}
        {step==="interests"&&(
          <div className="acard su">
            <div style={{fontSize:10,color:"#555",letterSpacing:.5,marginBottom:4}}>STEP 2 OF 2</div>
            <div style={{fontSize:17,fontWeight:700,color:"#f0ece4",marginBottom:4}}>What do you love watching?</div>
            <div style={{fontSize:11,color:"#666",marginBottom:16,lineHeight:1.5}}>Pick genres to personalise your feed.</div>
            <div className="intgrid">{INTEREST_LIST.map(g=><div key={g} className={`intpill ${interests.includes(g)?"sel":""}`} onClick={()=>toggleInt(g)}>{g}</div>)}</div>
            <button className="bmain" onClick={()=>setStep("app")} disabled={interests.length===0}>{interests.length===0?"Select at least one genre":`Continue with ${interests.length} genre${interests.length>1?"s":""}`}</button>
            <div className="afoot"><span onClick={()=>setStep("app")}>Skip for now →</span></div>
          </div>
        )}
        {/* LOGIN / SIGNUP */}
        {(step==="login"||step==="signup")&&(
          <div className="acard su">
            <div className="alogo">Cinema to Watch</div>
            <div className="atag">Movie recommendation engine powered by Machine Learning</div>
            <div className="atabs">
              {["login","signup"].map(t=>(
                <div key={t} className={`atab ${authTab===t?"active":""}`} onClick={()=>{setAuthTab(t);setStep(t);}}>{t==="login"?"Sign In":"Create Account"}</div>
              ))}
            </div>
            {authTab==="login"?(
              <>
                <div className="fields">
                  <div className="field"><label>Email address</label><input type="email" placeholder="you@example.com" value={loginEmail} onChange={e=>setLE(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doLoginEmail()}/></div>
                  <div className="field"><label>Password</label><input type="password" placeholder="••••••••" value={loginPass} onChange={e=>setLP(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doLoginEmail()}/></div>
                </div>
                <button className="bmain" onClick={doLoginEmail}>Sign In</button>
                <div className="or-d">or</div>
                <button className="bgoogle" onClick={()=>doGoogle("login")}><GIcon/>Continue with Google</button>
                <div className="afoot">No account? <span onClick={()=>{setAuthTab("signup");setStep("signup");}}>Sign up free</span></div>
              </>
            ):(
              <>
                <div className="fields">
                  <div className="field"><label>Full name</label><input type="text" placeholder="Your name" value={signupName} onChange={e=>setSN(e.target.value)}/></div>
                  <div className="field"><label>Email address</label><input type="email" placeholder="you@example.com" value={signupEmail} onChange={e=>setSE(e.target.value)}/></div>
                  <div className="field"><label>Password (min. 8 characters)</label><input type="password" placeholder="••••••••" value={signupPass} onChange={e=>setSP(e.target.value)}/></div>
                </div>
                <button className="bmain" onClick={doSignupEmail} disabled={!signupName||!signupEmail||signupPass.length<8}>Create Account — Verify Email</button>
                <div className="or-d">or</div>
                <button className="bgoogle" onClick={()=>doGoogle("signup")}><GIcon/>Sign up with Google</button>
                <div className="afoot">Have an account? <span onClick={()=>{setAuthTab("login");setStep("login");}}>Sign in</span></div>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );

  /* ══════════════════════════════════════════
     MAIN APP RENDER
  ══════════════════════════════════════════ */
  const NAV = [
    {id:"Home",             icon:"home"},
    {id:"For You",          icon:"sparkle"},
    {id:"Recommendations",  icon:"search"},
    {id:"History",          icon:"clock"},
    {id:"Explore Data",     icon:"chart"},
    {id:"Revenue Predictor",icon:"money"},
    {id:"Profile",          icon:"user"},
    {id:"Contact",          icon:"mail"},
  ];

  return (
    <>
      <style>{CSS}</style>
      {showGPicker && <GooglePicker/>}
      {selMovie && (
        <MovieModal
          movie={selMovie}
          watchedIds={watchedIds}
          onClose={()=>setSelMovie(null)}
          onWatch={m=>{markWatch(m); setSelMovie({...m});}}
          similar={getSimilar(selMovie)}
          onOpenMovie={m=>{ setSelMovie(null); setTimeout(()=>openMovie(m),80); }}
        />
      )}
      <div className="app">
        {/* SIDEBAR */}
        <div className="sb">
          <div className="sb-logo">Cinema to Watch</div>
          <div className="sb-sub">ML-powered movie recommendations</div>
          <div className="sb-lbl">Navigation</div>
          {NAV.map(({id,icon})=>(
            <div key={id} className={`nav ${page===id?"active":""}`} onClick={()=>{setPage(id);setSearched(false);}}>
              <Ico n={icon} size={14} color={page===id?"#e0a84b":"#555"} sw={page===id?2.2:1.8}/>{id}
            </div>
          ))}
          <hr className="sb-hr"/>
          <div className="uchip">
            <div className="av">{user?.avatar}</div>
            <div style={{overflow:"hidden"}}><div className="uname">{user?.name}</div><div className="uemail">{user?.email}</div></div>
          </div>
          <div className="sout" onClick={doLogout}>Sign out</div>
        </div>

        {/* MAIN CONTENT */}
        <div className="main">

          {/* ── HOME ── */}
          {page==="Home"&&(
            <div className="fi">
              {/* HERO */}
              <div className="hero-bg" key={heroKey}>
                <div style={{position:"absolute",inset:0,background:`linear-gradient(135deg, hsl(${heroIdx*40+20},40%,8%), hsl(${heroIdx*40+50},30%,12%))`,animation:"heroShift 12s ease-in-out infinite alternate"}}/>
                <div className="hero-grad"/>
                <div className="hero-content" style={{animation:"slideLeft .8s ease forwards"}}>
                  <div style={{display:"flex",gap:6,marginBottom:10,flexWrap:"wrap"}}>
                    {heroMovie.genres.map(g=><span key={g} style={{padding:"3px 10px",borderRadius:20,fontSize:11,background:"rgba(224,168,75,.2)",border:"1px solid rgba(224,168,75,.4)",color:"#e0a84b",fontWeight:700}}>{g}</span>)}
                  </div>
                  <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:52,color:"#f0ece4",lineHeight:.95,letterSpacing:1,marginBottom:10}}>{heroMovie.title}</div>
                  <div style={{fontSize:12,color:"#aaa",marginBottom:6,fontStyle:"italic"}}>{heroMovie.tagline}</div>
                  <div style={{fontSize:12,color:"#777",marginBottom:16}}>{heroMovie.year} · {Math.floor(heroMovie.runtime/60)}h {heroMovie.runtime%60}m · ⭐ {heroMovie.rating}</div>
                  <p style={{fontSize:13,color:"#888",lineHeight:1.7,marginBottom:20,maxWidth:440}}>{heroMovie.overview}</p>
                  <div style={{display:"flex",gap:10}}>
                    <button onClick={()=>openMovie(heroMovie)} style={{background:"#e0a84b",color:"#0c0c0c",border:"none",borderRadius:8,padding:"11px 24px",fontWeight:700,fontSize:13,cursor:"pointer",fontFamily:"'DM Sans',sans-serif",display:"flex",alignItems:"center",gap:8}}>
                      <Ico n="play" size={14} color="#0c0c0c" sw={2.5}/> View Details
                    </button>
                    <button onClick={()=>markWatch(heroMovie)} style={{background:"rgba(255,255,255,.08)",color:"#f0ece4",border:"1px solid rgba(255,255,255,.15)",borderRadius:8,padding:"11px 20px",fontWeight:600,fontSize:13,cursor:"pointer",fontFamily:"'DM Sans',sans-serif"}}>
                      {watchedIds.includes(heroMovie.id)?"✓ Watched":"+ Watchlist"}
                    </button>
                  </div>
                </div>
                {/* hero poster preview - right side, vertically centered, clear of text */}
                <div style={{position:"absolute",right:40,top:"50%",transform:"translateY(-50%)",width:120,borderRadius:10,overflow:"hidden",border:"2px solid rgba(255,255,255,.1)",boxShadow:"0 20px 60px rgba(0,0,0,.8)",opacity:.85,pointerEvents:"none"}}>
                  <Poster movieId={heroMovie.id} title={heroMovie.title}/>
                </div>
                {/* dots */}
                <div style={{position:"absolute",bottom:18,right:24,display:"flex",gap:6}}>
                  {MOVIES.slice(0,4).map((_,i)=>(
                    <div key={i} onClick={()=>{setHeroIdx(i);setHeroKey(k=>k+1);}} style={{width:i===heroIdx?22:7,height:7,borderRadius:4,background:i===heroIdx?"#e0a84b":"rgba(255,255,255,.2)",cursor:"pointer",transition:"all .3s"}}/>
                  ))}
                </div>
              </div>

              <div style={{padding:"24px 28px"}}>
                {/* TRENDING */}
                <div className="srow" style={{marginBottom:14}}>
                  <div style={{display:"flex",alignItems:"center",gap:8}}><Ico n="fire" size={16} color="#e0a84b"/><span className="sh" style={{fontSize:15}}>Trending Right Now</span></div>
                  <button onClick={fetchTrending} style={{background:"transparent",border:"1px solid #2a2a2a",borderRadius:6,padding:"4px 10px",fontSize:10,color:"#666",cursor:"pointer",fontFamily:"'DM Sans',sans-serif",display:"flex",alignItems:"center",gap:5}}>
                    <Ico n="trending" size={11} color="#666"/>Refresh
                  </button>
                </div>
                {trendLoad?(
                  <div style={{display:"flex",gap:12,overflow:"hidden",paddingBottom:8}}>
                    {Array.from({length:6}).map((_,i)=>(
                      <div key={i} style={{minWidth:145,flexShrink:0}}>
                        <div className="skel" style={{width:"100%",aspectRatio:"2/3",borderRadius:8,marginBottom:6}}/>
                        <div className="skel" style={{height:11,width:"80%",borderRadius:4,marginBottom:4}}/>
                        <div className="skel" style={{height:9,width:"50%",borderRadius:4}}/>
                      </div>
                    ))}
                  </div>
                ):trending.length>0?(
                  <div className="scroll-row">
                    {trending.map((m,i)=>(
                      <div key={i} className="row-card">
                        <div style={{background:"#141414",border:"1px solid #1e1e1e",borderRadius:9,overflow:"hidden",cursor:"pointer",transition:"all .2s"}}
                          onMouseEnter={e=>{e.currentTarget.style.borderColor="#e0a84b";e.currentTarget.style.transform="translateY(-3px)";}}
                          onMouseLeave={e=>{e.currentTarget.style.borderColor="#1e1e1e";e.currentTarget.style.transform="";}}
                          onClick={()=>{ const local=MOVIES.find(x=>x.title.toLowerCase()===m.title.toLowerCase()); if(local) openMovie(local); }}>
                          <div style={{background:`linear-gradient(135deg,hsl(${i*40},30%,10%),hsl(${i*40+20},20%,16%))`,aspectRatio:"2/3",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",padding:12,position:"relative"}}>
                            <div style={{position:"absolute",top:7,left:7,background:"rgba(0,0,0,.7)",color:"#e0a84b",fontSize:9,fontWeight:800,padding:"2px 6px",borderRadius:4,border:"1px solid rgba(224,168,75,.3)"}}>{m.year}</div>
                            <Ico n="film" size={32} color="#222"/>
                            <div style={{fontSize:10,color:"#333",marginTop:8,textAlign:"center",lineHeight:1.3,padding:"0 4px"}}>{m.title}</div>
                          </div>
                          <div style={{padding:"9px 10px 11px"}}>
                            <div style={{fontSize:11,fontWeight:700,color:"#f0ece4",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",marginBottom:3}}>{m.title}</div>
                            <div style={{fontSize:9,color:"#555",marginBottom:4}}>{m.year} · ⭐ {m.rating}</div>
                            <div style={{display:"flex",gap:3,flexWrap:"wrap"}}>
                              {(m.genres||[]).slice(0,1).map(g=><span key={g} style={{fontSize:9,color:"#666",background:"#1e1e1e",padding:"2px 6px",borderRadius:20}}>{g}</span>)}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ):(
                  <div style={{background:"#141414",border:"1px solid #1e1e1e",borderRadius:10,padding:"18px 20px",fontSize:12,color:"#555",display:"flex",alignItems:"center",gap:10}}>
                    <Ico n="film" size={16} color="#333"/>Click Refresh to load trending movies via AI
                  </div>
                )}

                {/* TOP RATED */}
                <div className="srow" style={{marginTop:28,marginBottom:14}}>
                  <div style={{display:"flex",alignItems:"center",gap:8}}><Ico n="star" size={15} color="#e0a84b"/><span className="sh" style={{fontSize:15}}>Top Rated in Library</span></div>
                </div>
                <div className="scroll-row">
                  {[...MOVIES].sort((a,b)=>b.rating-a.rating).map(m=>(
                    <div key={m.id} className="row-card"><MovieCard movie={m} watchedIds={watchedIds} onOpen={openMovie} onWatch={markWatch} compact/></div>
                  ))}
                </div>

                {/* SCI-FI SPOTLIGHT */}
                <div style={{marginTop:28,marginBottom:14}} className="srow">
                  <div style={{display:"flex",alignItems:"center",gap:8}}><Ico n="film" size={15} color="#e0a84b"/><span className="sh" style={{fontSize:15}}>Sci-Fi & Thrillers</span></div>
                </div>
                <div className="g5">
                  {MOVIES.filter(m=>m.genres.some(g=>["Sci-Fi","Thriller"].includes(g))).slice(0,5).map(m=>(
                    <MovieCard key={m.id} movie={m} watchedIds={watchedIds} onOpen={openMovie} onWatch={markWatch} compact/>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── FOR YOU ── */}
          {page==="For You"&&(
            <div className="fi page">
              <div className="ptitle">Hi {user?.name?.split(" ")[0]}, welcome</div>
              <div className="psub">Personalised picks based on your real-time activity and taste profile.</div>
              <div style={{background:"linear-gradient(135deg,#1a1200,#141414)",border:"1px solid #2e2200",borderRadius:11,padding:"16px 20px",marginBottom:18,display:"flex",alignItems:"center",gap:13}}>
                <Ico n="sparkle" size={22} color="#e0a84b"/>
                <div>
                  <div style={{fontSize:13,fontWeight:700,color:"#e0a84b",marginBottom:3}}>Your Taste Profile</div>
                  <div style={{fontSize:11,color:"#666",lineHeight:1.5}}>{activity.length===0&&interests.length===0?"Start watching or searching to build your profile.":`${activity.length} activit${activity.length===1?"y":"ies"} · Interests: ${interests.slice(0,4).join(", ")||"None set"}`}</div>
                </div>
              </div>
              <div className="srow"><div className="sh">Recommended For You</div><div style={{fontSize:10,color:"#333"}}>Click any movie for full details</div></div>
              {personalized.length===0
                ?<div className="empty">You've seen everything! Great taste.</div>
                :<div className="g5">{personalized.slice(0,10).map((m,i)=><div key={m.id} className="fi" style={{animationDelay:`${i*.04}s`}}><MovieCard movie={m} watchedIds={watchedIds} onOpen={openMovie} onWatch={markWatch}/></div>)}</div>
              }
              {watchedIds.length>0&&<>
                <div style={{marginTop:22}} className="srow"><div className="sh">Recently Watched</div></div>
                <div className="hlist">
                  {activity.filter(a=>a.action==="Watched").slice(0,4).map((a,i)=>(
                    <div key={i} className="hitem fi" style={{animationDelay:`${i*.03}s`}} onClick={()=>{const m=MOVIES.find(x=>x.id===a.mid);if(m)openMovie(m);}}>
                      <div style={{width:36,height:54,flexShrink:0,borderRadius:4,overflow:"hidden"}}><Poster movieId={a.mid} title={a.title}/></div>
                      <div style={{flex:1}}><div style={{fontSize:12,fontWeight:600,color:"#f0ece4",marginBottom:2}}>{a.title}</div><div style={{fontSize:10,color:"#555"}}>{a.genres?.slice(0,2).join(", ")}</div></div>
                      <span className="tag tw">Watched</span>
                    </div>
                  ))}
                </div>
              </>}
            </div>
          )}

          {/* ── RECOMMENDATIONS ── */}
          {page==="Recommendations"&&(
            <div className="fi page">
              <div className="ptitle">Recommendations</div>
              <div className="psub">Search any movie · click any card for full details.</div>
              <div className="sbar">
                <div className="ig"><div className="ilbl">Movie title</div>
                  <input className="inp" placeholder="e.g. Inception, Joker, Dune..." value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==="Enter"&&doSearch()}/>
                </div>
                <button className="btn" onClick={doSearch}>Search</button>
              </div>
              <div className="ilbl" style={{marginBottom:6}}>Filter by genre</div>
              <div className="gpills">{GENRE_LIST.map(g=><div key={g} className={`gp ${selGenres.includes(g)?"sel":""}`} onClick={()=>toggleG(g)}>{g}</div>)}</div>
              {searched
                ?<div className="fi"><div className="sbanner">Results for "<strong>{query}</strong>" · {filteredMovies.length} found</div>
                  <div className="g5">{filteredMovies.map(m=><MovieCard key={m.id} movie={m} watchedIds={watchedIds} onOpen={openMovie} onWatch={markWatch}/>)}</div>
                  {filteredMovies.length===0&&<div className="empty">No results. Try another title.</div>}
                </div>
                :<div className="g5">{MOVIES.slice(0,10).map(m=><MovieCard key={m.id} movie={m} watchedIds={watchedIds} onOpen={openMovie} onWatch={markWatch}/>)}</div>
              }
            </div>
          )}

          {/* ── HISTORY ── */}
          {page==="History"&&(
            <div className="fi page">
              <div className="ptitle">Activity & History</div>
              <div className="psub">Real-time log — drives personalised recommendations.</div>
              <div className="g3" style={{marginBottom:18}}>
                <div className="mc"><div className="mv">{activity.filter(a=>a.action==="Watched").length}</div><div className="ml">Watched</div></div>
                <div className="mc"><div className="mv">{activity.filter(a=>a.action==="Searched").length}</div><div className="ml">Searched</div></div>
                <div className="mc"><div className="mv">{activity.length}</div><div className="ml">Total</div></div>
              </div>
              <div className="tabs">{["All","Watched","Searched","Viewed"].map(t=><div key={t} className={`tab ${histTab===t?"active":""}`} onClick={()=>setHistTab(t)}>{t}</div>)}</div>
              {activity.filter(a=>histTab==="All"||a.action===histTab).length===0
                ?<div className="empty">No {histTab.toLowerCase()} activity yet.</div>
                :<div className="hlist">{activity.filter(a=>histTab==="All"||a.action===histTab).map((a,i)=>(
                  <div key={i} className="hitem fi" style={{animationDelay:`${i*.03}s`}} onClick={()=>{const m=MOVIES.find(x=>x.id===a.mid);if(m)openMovie(m);}}>
                    <div style={{width:36,height:54,flexShrink:0,borderRadius:4,overflow:"hidden"}}><Poster movieId={a.mid} title={a.title}/></div>
                    <div style={{flex:1}}><div style={{fontSize:12,fontWeight:600,color:"#f0ece4",marginBottom:2}}>{a.title}</div><div style={{fontSize:10,color:"#555"}}>{a.genres?.slice(0,2).join(", ")}</div></div>
                    <span className={`tag ${a.action==="Watched"?"tw":a.action==="Searched"?"ts2":"tv"}`}>{a.action}</span>
                    <div style={{fontSize:10,color:"#333",whiteSpace:"nowrap"}}>{a.time}</div>
                  </div>
                ))}</div>
              }
            </div>
          )}

          {/* ── EXPLORE DATA ── */}
          {page==="Explore Data"&&(
            <div className="fi page">
              <div className="ptitle">Explore Dataset</div>
              <div className="psub">Visual insights from 4,803 TMDB movies.</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:18}}>
                {[{v:"4,803",l:"Movies"},{v:"20",l:"Genres"},{v:"6.9",l:"Avg Rating"},{v:"1916–2017",l:"Years"}].map((m,i)=>(
                  <div className="mc" key={i}><div className="mv" style={{fontSize:m.v.length>5?18:24}}>{m.v}</div><div className="ml">{m.l}</div></div>
                ))}
              </div>
              <div className="tabs">{["Genres","Ratings","Over Time","Clusters"].map(t=><div key={t} className={`tab ${exTab===t?"active":""}`} onClick={()=>setExTab(t)}>{t}</div>)}</div>
              {exTab==="Genres"&&<div className="fi chart">{GENRE_STATS.map((g,i)=><div className="brow" key={i}><div className="blbl">{g.l}</div><div className="btrack"><div className="bfill" style={{width:`${g.p}%`,background:"#e0a84b"}}>{g.n}</div></div><div style={{fontSize:9,color:"#555",width:28}}>{g.p}%</div></div>)}</div>}
              {exTab==="Ratings"&&<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:11}}>{[{d:[2,8,18,42,85,120,145,130,90,45,20,8],c:"#e0a84b",t:"Rating Distribution",lx:"1.0",rx:"10.0"},{d:[5,15,40,95,145,130,90,55,30,18,10,4],c:"#5b8fd4",t:"Runtime Distribution",lx:"30m",rx:"240m"}].map((ch,i)=><div key={i} className="fi chart"><div className="ctitle">{ch.t}</div><div style={{display:"flex",alignItems:"flex-end",gap:3,height:80}}>{ch.d.map((h,j)=><div key={j} style={{flex:1,height:`${(h/145)*100}%`,background:ch.c,borderRadius:"2px 2px 0 0",opacity:.85}}/>)}</div><div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:9,color:"#444"}}><span>{ch.lx}</span><span>{ch.rx}</span></div></div>)}</div>}
              {exTab==="Over Time"&&<div className="fi chart"><div className="ctitle">Movies Per Year (1990–2017)</div><div style={{display:"flex",alignItems:"flex-end",gap:2,height:100}}>{[18,22,28,32,38,42,50,55,60,70,85,95,110,125,140,155,165,170,175,180,185,190,195,200,210,215,220,225].map((h,i)=><div key={i} style={{flex:1,height:`${(h/225)*100}%`,background:`hsl(${38+i*1.5},70%,${45+i*.3}%)`,borderRadius:"2px 2px 0 0"}}/>)}</div><div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:9,color:"#444"}}><span>1990</span><span>2000</span><span>2010</span><span>2017</span></div></div>}
              {exTab==="Clusters"&&<div className="fi chart"><div className="ctitle">PCA — 2D Cluster Projection</div><svg width="100%" viewBox="0 0 500 255"><rect width="500" height="255" fill="#111" rx="8"/>{[{cx:120,cy:72,c:"#e0a84b",l:"Blockbusters"},{cx:275,cy:56,c:"#5b8fd4",l:"Indie"},{cx:398,cy:102,c:"#7ecb8a",l:"Animated"},{cx:172,cy:185,c:"#c97ecc",l:"Drama"},{cx:332,cy:192,c:"#e07b5b",l:"Thriller"},{cx:68,cy:202,c:"#5bd4c9",l:"Docs"},{cx:452,cy:230,c:"#d4d45b",l:"Western"}].map((c,i)=><g key={i}>{Array.from({length:20}).map((_,j)=><circle key={j} cx={c.cx+(Math.sin(j*17+i)*37)} cy={c.cy+(Math.cos(j*13+i)*32)} r={2.5} fill={c.c} opacity={.55}/>)}<text x={c.cx} y={c.cy} textAnchor="middle" fill={c.c} fontSize="9" fontFamily="DM Sans" fontWeight="700">{c.l}</text></g>)}</svg></div>}
            </div>
          )}

          {/* ── REVENUE PREDICTOR ── */}
          {page==="Revenue Predictor"&&(
            <div className="fi page">
              <div className="ptitle">Revenue Predictor</div>
              <div className="psub">Estimate box office using the Random Forest model.</div>
              <div className="g3" style={{marginBottom:18}}>
                <div className="mc"><div className="mv">0.72</div><div className="ml">R² Score</div></div>
                <div className="mc"><div className="mv">$41M</div><div className="ml">Mean Abs. Error</div></div>
                <div className="mc"><div className="mv">RF</div><div className="ml">Model</div></div>
              </div>
              <div className="pgrid">
                <div className="panel"><div className="ptl">Feature Importance</div>
                  {[{l:"Budget",p:82,c:"#e0a84b"},{l:"Popularity",p:61,c:"#5b8fd4"},{l:"Vote Count",p:47,c:"#7ecb8a"},{l:"Runtime",p:28,c:"#c97ecc"},{l:"Vote Avg",p:22,c:"#e07b5b"}].map((f,i)=>(
                    <div className="brow" key={i}><div className="blbl">{f.l}</div><div className="btrack"><div className="bfill" style={{width:`${f.p}%`,background:f.c}}>{f.p}%</div></div></div>
                  ))}
                </div>
                <div className="panel"><div className="ptl">Predict Revenue</div>
                  {[{l:"Budget",v:budget,set:setBudget,min:1,max:300,d:`$${budget}M`},{l:"Popularity",v:pop,set:setPop,min:1,max:300,d:pop},{l:"Runtime (min)",v:runtime2,set:setRuntime,min:60,max:240,d:`${runtime2}m`},{l:"Vote Avg",v:rat2*10,set:v=>setRat(v/10),min:10,max:100,d:rat2.toFixed(1)},{l:"Vote Count",v:votes2/100,set:v=>setVotes(v*100),min:1,max:200,d:`${(votes2/1000).toFixed(1)}k`}].map((s,i)=>(
                    <div key={i} style={{marginBottom:10}}><div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}><span style={{fontSize:11,color:"#555"}}>{s.l}</span><span style={{fontSize:11,color:"#e0a84b",fontWeight:700}}>{s.d}</span></div><input type="range" min={s.min} max={s.max} value={s.v} onChange={e=>{s.set(Number(e.target.value));setShowRev(false);}}/></div>
                  ))}
                  <button className="btn" style={{width:"100%",marginTop:8}} onClick={()=>setShowRev(true)}>Predict Revenue</button>
                  {showRev&&<div style={{background:"#1a1600",border:"1px solid #2e2200",borderRadius:8,padding:"14px",textAlign:"center",marginTop:12}} className="fi"><div style={{fontSize:10,color:"#666",marginBottom:3}}>Estimated Revenue</div><div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:32,color:"#e0a84b",letterSpacing:1}}>${(estRev(budget,pop,runtime2,rat2,votes2)/1e6).toFixed(0)}M</div></div>}
                </div>
              </div>
            </div>
          )}

          {/* ── PROFILE ── */}
          {page==="Profile"&&(
            <div className="fi page">
              <div className="ptitle">Your Profile</div>
              <div className="psub">Account settings, taste preferences, and watch history.</div>
              <div style={{display:"grid",gridTemplateColumns:"260px 1fr",gap:14}}>
                <div>
                  <div style={{background:"#141414",border:"1px solid #1e1e1e",borderRadius:11,padding:22,textAlign:"center",marginBottom:12}}>
                    <div style={{width:64,height:64,borderRadius:"50%",background:"#e0a84b",display:"flex",alignItems:"center",justifyContent:"center",fontSize:22,fontWeight:800,color:"#0c0c0c",margin:"0 auto 10px"}}>{user?.avatar}</div>
                    <div style={{fontFamily:"'Bebas Neue',sans-serif",fontSize:20,color:"#f0ece4",letterSpacing:1,marginBottom:3}}>{user?.name}</div>
                    <div style={{fontSize:10,color:"#555",marginBottom:14}}>{user?.email}</div>
                    <div style={{display:"flex",justifyContent:"space-around",marginBottom:14}}>
                      {[{v:watchedIds.length,l:"Watched"},{v:activity.length,l:"Activities"},{v:interests.length,l:"Interests"}].map((s,i)=>(
                        <div key={i}><div style={{fontSize:18,fontWeight:800,color:"#e0a84b"}}>{s.v}</div><div style={{fontSize:9,color:"#555"}}>{s.l}</div></div>
                      ))}
                    </div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:5,justifyContent:"center"}}>
                      {interests.map(t=><span key={t} style={{padding:"2px 8px",borderRadius:20,fontSize:9,background:"#1a1600",border:"1px solid #2e2200",color:"#e0a84b"}}>{t}</span>)}
                    </div>
                  </div>
                  <div className="panel"><div className="ptl">Update Interests</div><div style={{display:"flex",flexWrap:"wrap",gap:5}}>{INTEREST_LIST.map(g=><div key={g} className={`intpill ${interests.includes(g)?"sel":""}`} style={{padding:"5px 8px"}} onClick={()=>toggleInt(g)}>{g}</div>)}</div></div>
                </div>
                <div>
                  <div className="panel" style={{marginBottom:12}}>
                    <div className="ptl">Watch History ({watchedIds.length})</div>
                    {watchedIds.length===0
                      ?<div style={{fontSize:11,color:"#3a3a3a",padding:"6px 0"}}>No movies watched yet.</div>
                      :<div style={{display:"flex",flexDirection:"column",gap:7}}>{MOVIES.filter(m=>watchedIds.includes(m.id)).map(m=>(
                        <div key={m.id} style={{display:"flex",alignItems:"center",gap:10,padding:"7px 0",borderBottom:"1px solid #1a1a1a",cursor:"pointer"}} onClick={()=>openMovie(m)}>
                          <div style={{width:30,height:45,flexShrink:0,borderRadius:4,overflow:"hidden"}}><Poster movieId={m.id} title={m.title}/></div>
                          <div style={{flex:1}}><div style={{fontSize:11,color:"#d0ccc5",fontWeight:600}}>{m.title}</div><div style={{fontSize:10,color:"#444"}}>{m.year} · ⭐ {m.rating}</div></div>
                          <span className="tag tw">Watched</span>
                        </div>
                      ))}</div>
                    }
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── CONTACT ── */}
          {page==="Contact"&&(
            <div className="fi page">
              <div className="ptitle">Contact Us</div>
              <div className="psub">Have a question, suggestion, or issue? Reach out to the admin team.</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 340px",gap:20}}>
                <div className="panel">
                  <div className="ptl">Send a Message</div>
                  {contactSent?(
                    <div style={{textAlign:"center",padding:"30px 0"}}>
                      <div style={{fontSize:32,marginBottom:12}}>✅</div>
                      <div style={{fontSize:15,fontWeight:700,color:"#f0ece4",marginBottom:6}}>Message sent!</div>
                      <div style={{fontSize:12,color:"#666",marginBottom:18}}>We'll get back to you within 24 hours.</div>
                      <button className="btn" onClick={()=>{setContact(false);setCName("");setCEmail("");setCMsg("");}}>Send another</button>
                    </div>
                  ):(
                    <div className="cform">
                      <input type="text" placeholder="Your name" value={cName} onChange={e=>setCName(e.target.value)}/>
                      <input type="email" placeholder="Your email address" value={cEmail} onChange={e=>setCEmail(e.target.value)}/>
                      <textarea rows={5} placeholder="Your message..." value={cMsg} onChange={e=>setCMsg(e.target.value)}/>
                      <button className="btn" style={{width:"100%"}} onClick={()=>{if(cName&&cEmail&&cMsg)setContact(true);}}>Send Message</button>
                    </div>
                  )}
                </div>
                <div style={{display:"flex",flexDirection:"column",gap:12}}>
                  {[{icon:"mail",title:"Email Support",val:"admin@cinematowatch.com",sub:"Replies within 24 hours"},
                    {icon:"user",title:"Admin Team",val:"Cinema to Watch Team",sub:"Mon–Fri, 9am–6pm PKT"},
                    {icon:"clock",title:"Response Time",val:"Within 24 hours",sub:"For urgent issues, email directly"},
                  ].map((c,i)=>(
                    <div key={i} style={{background:"#141414",border:"1px solid #1e1e1e",borderRadius:10,padding:"16px 18px",display:"flex",gap:13,alignItems:"flex-start"}}>
                      <div style={{width:36,height:36,borderRadius:8,background:"#1a1600",border:"1px solid #2e2200",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}><Ico n={c.icon} size={16} color="#e0a84b"/></div>
                      <div><div style={{fontSize:12,fontWeight:700,color:"#ccc",marginBottom:3}}>{c.title}</div><div style={{fontSize:13,color:"#e0a84b",fontWeight:500,marginBottom:2}}>{c.val}</div><div style={{fontSize:10,color:"#555"}}>{c.sub}</div></div>
                    </div>
                  ))}
                  <div style={{background:"#141414",border:"1px solid #1e1e1e",borderRadius:10,padding:"16px 18px"}}>
                    <div style={{fontSize:11,fontWeight:700,color:"#555",textTransform:"uppercase",letterSpacing:1,marginBottom:12}}>Find us online</div>
                    {[{p:"GitHub",h:"@cinema-to-watch"},{p:"Twitter / X",h:"@cinematowatch"},{p:"LinkedIn",h:"Cinema to Watch"}].map((s,i)=>(
                      <div key={i} style={{display:"flex",justifyContent:"space-between",padding:"7px 0",borderBottom:i<2?"1px solid #1e1e1e":"none",alignItems:"center"}}>
                        <span style={{fontSize:11,color:"#666"}}>{s.p}</span>
                        <span style={{fontSize:11,color:"#e0a84b",fontWeight:500,cursor:"pointer"}}>{s.h}</span>
                      </div>
                    ))}
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
