import {type FastifyInstance, type FastifyServerOptions} from "fastify";

/**
 * A plugin that provide encapsulated routes
 * @param {FastifyInstance} fastify encapsulated fastify instance
 * @param {Object} options plugin options, refer to https://fastify.dev/docs/latest/Reference/Plugins/#plugin-options
 */
async function routes (fastify: FastifyInstance, options: FastifyServerOptions) {
    // const collection = fastify.mongo.db.collection('test_collection')

    // fastify.get('/about', async (request, reply) => {
    //     const htmlResponse = `
    //   <h2 class="text-3xl mb-4">About Us</h2>
    //   <p>This is a sample Single Page Application (SPA) built to demonstrate dynamic content loading.
    //   Here, you can find various sections, like About and Contact, which are loaded when you click the respective buttons.</p>
    // `;
    //     // Retourner la réponse HTML
    //     return reply.type('text/html').send(htmlResponse);
    // });
    //
    // fastify.get('/contact', async (request, reply) => {
    //     const htmlResponse = `
    //   <h2 class="text-3xl mb-4">Contact Us</h2>
    //   <p>If you want to reach out to us, you can contact us through email or phone. We'd love to hear from you!</p>
    //   <p>Email: example@example.com</p>
    //   <p>Phone: (123) 456-7890</p>
    // `;
    //     // Retourner la réponse HTML
    //     return reply.type('text/html').send(htmlResponse);
    // });

    // fastify.get('/animals', async (request, reply) => {
    //     const result = await collection.find().toArray()
    //     if (result.length === 0) {
    //         throw new Error('No documents found')
    //     }
    //     return result
    // })

    // fastify.get('/animals/:animal', async (request, reply) => {
    //     const result = await collection.findOne({ animal: request.params.animal })
    //     if (!result) {
    //         throw new Error('Invalid value')
    //     }
    //     return result
    // })

    const animalBodyJsonSchema = {
        type: 'object',
        required: ['animal'],
        properties: {
            animal: { type: 'string' },
        },
    }

    const schema = {
        body: animalBodyJsonSchema,
    }

    // fastify.post('/animals', { schema }, async (request, reply) => {
    //     // we can use the `request.body` object to get the data sent by the client
    //     const result = await collection.insertOne({ animal: request.body.animal })
    //     return result
    // })
}
export default routes