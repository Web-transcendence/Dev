import Fastify from 'fastify'
import {existsSync, readFileSync} from "node:fs";
import {join} from "node:path";

import dbConnector from './our-db-connector.js'
import firstRoute from './our-first-route.js'
import {env} from "./env.js";
import {z} from "zod";

/**
 * @type {import('fastify').FastifyInstance} Instance of Fastify
 */

// Load SSL certificates
// const httpsOptions = {
//     https: {
//         key: readFileSync('./key.pem'),      // Private key
//         cert: readFileSync('./cert.pem')     // Certificate
//     },
//     logger: true
// };

// const fastify = Fastify(httpsOptions);
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

fastify.get("/:file", (req, res) => {
    const zParams = z.object({
        file: z.string()
    })
    const { success, error, data } = zParams.safeParse(req.params);
    if (!success) {
        res.raw.writeHead(400);
        res.raw.write(error);
        res.raw.end();
        return ;
    }
    let { file } = data;
    if (file.split('.').length === 1) {
        file = `${file}.html`;
    }
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, file);
    if (!existsSync(pagePath))
    {
        res.raw.writeHead(404);
        res.raw.end();
        return ;
    }
    const readFile = readFileSync(pagePath, 'utf8');
    res.raw.writeHead(200, { 'Content-Type': 'text/html' });
    res.raw.write(readFile);
    res.raw.end();
});

const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
fastify.get("/front.js", (req, res) => {
    const file = readFileSync(frontPath, 'utf8');
    res.raw.writeHead(200, { 'Content-Type': 'text/javascript' });
    res.raw.write(file);
    res.raw.end();
});

fastify.listen({ host: '0.0.0.0', port: 8443 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})