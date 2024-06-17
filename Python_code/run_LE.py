def rle_encode(data):
    encoding = []
    i = 0

    while i < len(data):
        # Count the number of occurrences of the current character
        count = 1
        while i + 1 < len(data) and data[i] == data[i + 1]:
            count += 1
            i += 1
        # Append the character and its count to the encoding
        encoding.append((data[i], count))
        i += 1

    return encoding

def rle_decode(encoded_data):
    decoded_data = []

    for char, count in encoded_data:
        decoded_data.append(char * count)

    return ''.join(decoded_data)

# Example usage
data = 'AAAABBBCCDAA'
print("Original Data:", data)

encoded_data = rle_encode(data)
print("Encoded Data:", encoded_data)

decoded_data = rle_decode(encoded_data)
print("Decoded Data:", decoded_data)
