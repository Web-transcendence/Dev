import Fastify, {FastifyReply, FastifyRequest} from "fastify";
import httpProxy from '@fastify/http-proxy';
import fastifyCookie from '@fastify/cookie';
import fastifyJwt from '@fastify/jwt';
import {readFileSync} from "node:fs";
import {join} from "node:path";
import fastifyStatic from "@fastify/static";
import {env} from "./env";
import {routes} from "./routes";

const httpsOptions = {
    https: {
        key: readFileSync(join(import.meta.dirname, '../../secure/key.pem')),      // Private key
        cert: readFileSync(join(import.meta.dirname, '../../secure/cert.pem'))     // Certificate
    },
};


const INTERNAL_PASSWORD = process.env.SECRET_KEY;

const app = Fastify(httpsOptions);

app.register(fastifyCookie);
app.register(fastifyJwt, {
    secret: INTERNAL_PASSWORD,
    cookie: {
        cookieName: 'token',
        signed: false
    }
});

async function authentificate (req: FastifyRequest, reply: FastifyReply) {
    if (req.url === "/user-management/login" || req.url === "/user-management/register" || req.url === "/user-management/auth/google" || req.url === "/user-management/2faVerify")
        return;
    try {
        const token = req.cookies.token;
        if (!token)
            return reply.status(401).send({ error: "Unauthorized - invalid token" });
        await req.jwt.verify(token)
    }
    catch (error) {
        return reply.status(401).send({ error: "Unauthorized - invalid token" });
    }
}

app.get('/authJWT', authentificate);

app.register(fastifyStatic, {
    root: join(import.meta.dirname, "..", "..", "public"),
    prefix: "/static/"
})

app.register(routes)

app.register(httpProxy, {
    upstream: 'http://match-server:4443',
    prefix: '/match-server',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://tower-defense:2246',
    prefix: '/tower-defense',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://user-management:5000',
    prefix: '/user-management',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://social:6500',
    prefix: '/social',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://tournament:7000',
    prefix: '/tournament',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://tower-defense:2246/ws',
    prefix: '/tower-defense/ws',
    websocket: true
});

app.register(httpProxy, {
    upstream: 'http://match-server:4443/ws',
    prefix: '/match-server/ws',
    websocket: true
});


app.get("/*", (req, res) => { // Route pour la page d'accueil
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    res.status(202).type('text/html').send(readFile);
});


app.listen({port: 4000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}, ${join(import.meta.dirname, '../secure/cert.pem')}`);
});