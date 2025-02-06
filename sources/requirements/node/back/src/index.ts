import Fastify from 'fastify'
import {readFileSync} from "node:fs";
import {join} from "node:path";

import dbConnector from './our-db-connector.js'
import firstRoute from './our-first-route.js'
import {env} from "./env.js";

/**
 * @type {import('fastify').FastifyInstance} Instance of Fastify
 */
const fastify = Fastify({
    logger: true
})
fastify.register(dbConnector)
fastify.register(firstRoute)

const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
fastify.get("/", (req, res) => {
    const file = readFileSync(pagePath, 'utf8');
    res.raw.writeHead(200, { 'Content-Type': 'text/html' });
    res.raw.write(file);
    res.raw.end();
});

const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
fastify.get("/front.js", (req, res) => {
    const file = readFileSync(frontPath, 'utf8');
    res.raw.writeHead(200, { 'Content-Type': 'text/javascript' });
    res.raw.write(file);
    res.raw.end();
});

fastify.listen({ host: '0.0.0.0', port: 8081 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})