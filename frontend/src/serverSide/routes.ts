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
    fastify.get("/pong.js", (req, res) => {
        const pongPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "pong.js");
        const file = readFileSync(pongPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    // ROAD OF TAG
    fastify.get('/part/about', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "about.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/connected', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "connected.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(htmlContent);
    })
    fastify.get('/part/logout', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "home.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(htmlContent);
    })
    fastify.get('/part/login', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "login.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(htmlContent);
    })
    fastify.get('/part/register', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "register.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/connect', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "register.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/contact', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "contact.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/profile', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "profile.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/home', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "home.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/towerDefense', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "towerDefense.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    // FAV ICON
    fastify.get('/images/favicon.ico', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "nobglogo.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/images/login.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "login.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/images/logout.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "logout.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/part/2fa', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "factor.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/pong', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "pong.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    //navigation bar, under page
    fastify.get('/part/pongMode', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "pongMode.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/tower', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "tower.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/tournaments', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "tournaments.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    // Png For Pong
    fastify.get('/assets/pong/ballup.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/ballup.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier ballup non trouvé");
        }
    });
    fastify.get('/assets/pong/bardown.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/bardown.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier bardown non trouvé");
        }
    });
    fastify.get('/assets/pong/barup.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/barup.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier barup non trouvé");
        }
    });
    fastify.get('/assets/pong/pong.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "pong/pong.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier pong non trouvé");
        }
    });
    fastify.get('/assets/docker_compose.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "docker_compose.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier docker_compose non trouvé");
        }
    });
    fastify.get('/assets/tower-defense/:filename', function (req, reply) {
        try {
            const { filename } = req.params;
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, 'tower-defense', filename);
            const file = readFileSync(frontPath);
            reply.code(224).type('image/png').send(file);
        } catch (error) {
            reply.code(404).send(`Fichier ${req.params.filename} non trouvé`);
        }
    });
    //Menu mode pong
    fastify.get('/images/vsia.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "vsia.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier vsia non trouvé");
        }
    });
    fastify.get('/images/remote.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "remote.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier remote non trouvé");
        }
    });
    fastify.get('/images/solo.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_IMG_PATH, "solo.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier solo non trouvé");
        }
    });
}