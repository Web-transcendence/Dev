import { FastifyInstance } from 'fastify';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { env } from './env.js';

export async function routes(fastify: FastifyInstance) {
    fastify.get("/front.js", (req, res) => {
        const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
        const file = readFileSync(frontPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    const htmlRoutes = [
        "about",
        "contact",
        "home",
        "shopDiscovery",
        "mentionsLegales",
    ];
    htmlRoutes.forEach(route => {
        fastify.get(`/part/${route}`, (req, reply) => {
            const filename = route === "logout" ? "home.html" : `${route}.html`;
            try {
                const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, filename);
                const html = readFileSync(fullPath, "utf8");
                reply.type("text/html").send(html);
            } catch (error) {
                reply.code(404).send(`Page ${filename} non trouvée`);
            }
        });
    });
    fastify.get('/part/:name', (req, reply) => {
        try {
            const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, 'index.html');
            const html = readFileSync(fullPath, 'utf8');
            reply.type('text/html').send(html);
        } catch (error) {
            reply.code(500).send('Erreur serveur : index.html introuvable');
        }
    });
    const pngImages = [
        "bag",
        "DeliveryMan",
        "Monitor",
        "NoFees",
        "Parteurs",
        "shopWeb",
        "shopFront",
        "StoreScreen",
        "ShopsScreen",
        "iPhoneFrameNo",
        "logoFonceV2",
        "logobigbwNobg",
        "MainPhonePage",
        "Vestimentaire",
    ];
    pngImages.forEach(name => {
        fastify.get(`/images/${name}.png`, (req, reply) => {
            try {
                const fullPath = join(import.meta.dirname, env.TRANS_IMG_PATH, `${name}.png`);
                const imageData = readFileSync(fullPath);

                reply.type("image/png").send(imageData);
            } catch (error) {
                reply.code(404).send(`Image ${name}.png non trouvée`);
            }
        });
    });
}