/**
 * BlockCipher.h
 *
 * A simple abstraction for the basic functionality of a block cipher engine.
 *
 * @author Paulo S. L. M. Barreto
 *
 * This software is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __BLOCKCIPHER_H
#define __BLOCKCIPHER_H

#ifndef USUAL_TYPES
#define USUAL_TYPES
typedef unsigned char   byte;
typedef unsigned long   uint;   /* assuming sizeof(uint) == 4 */
#endif /* USUAL_TYPES */

#define DIR_NONE    0

/**
 * Flag to setup the encryption key schedule.
 */
#define DIR_ENCRYPT 1

/**
 * Flag to setup the decryption key schedule.
 */
#define DIR_DECRYPT 2

/**
 * Flag to setup both key schedules (encryption/decryption).
 */
#define DIR_BOTH    (DIR_ENCRYPT | DIR_DECRYPT) /* both directions */

class BlockCipher {

public:

    /**
     * Block size in bits.
     */
    virtual uint blockBits() const = 0;

    /**
     * Block size in bytes.
     */
    virtual uint blockSize() const = 0;

    /**
     * Key size in bits.
     */
    virtual uint keyBits() const = 0;

    /**
     * Key size in bytes.
     */
    virtual uint keySize() const = 0;

    /**
     * Convert one data block from byte[] to uint[] representation.
     */
    virtual void byte2int(const byte *b, uint *i) = 0;

    /**
     * Convert one data block from int[] to byte[] representation.
     */
    virtual void int2byte(const uint *i, byte *b) = 0;

    /**
     * Setup the key schedule for encryption, decryption, or both.
     *
     * @param   cipherKey   the cipher key.
     * @param   keyBits     size of the cipher key in bits.
     * @param   direction   cipher direction (DIR_ENCRYPT, DIR_DECRYPT, or DIR_BOTH).
     */
    virtual void makeKey(const byte *cipherKey, uint keyBits, uint dir) = 0;

    /**
     * Encrypt exactly one block of plaintext.
     *
     * @param   pt          plaintext block.
     * @param   ct          ciphertext block.
     */
    virtual void encrypt(const uint *pt, uint *ct) = 0;

    /**
     * Decrypt exactly one block of ciphertext.
     *
     * @param   ct          ciphertext block.
     * @param   pt          plaintext block.
     */
    virtual void decrypt(const uint *ct, uint *pt) = 0;

};

#endif /* __BLOCKCIPHER_H */
