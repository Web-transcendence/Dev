import { routes } from './routes.js';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import {join} from "node:path";
import {env} from "./env.js";
import {readFileSync} from "node:fs";


const httpsOptions = {
    https: {
        key: readFileSync(join(import.meta.dirname, '../secure/key.pem')),      // Private key
        cert: readFileSync(join(import.meta.dirname, '../secure/cert.pem'))     // Certificate
    },
    logger: true
};

const fastify = Fastify(httpsOptions)

fastify.register(fastifyStatic, {
    root: join(import.meta.dirname, "..", "public"),
    prefix: "/static/"
})

fastify.get("/*", (req, res) => { // Route pour la page d'accueil
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    res.type('text/html').send(readFile);
});

fastify.register(routes)

fastify.listen({ host: '0.0.0.0', port: 4000 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now  listening on ${address}`)
})
